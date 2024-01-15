# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import random
import logging
import argparse
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from peft import LoraConfig, TaskType, get_peft_model, tuners, PeftModel
import bitsandbytes as bnb

from utils import NFQuantizer
from quantize.int_linear import QuantLinear

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        logging.info(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    logging.info("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    logging.info("=" * 10 + name + "=" * 10)
    logging.info(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                logging.info(
                    f"{name}, "
                    f"{param.shape}, "
                    f"{param.device}, "
                    f"{param.dtype}, "
                    f"{param.requires_grad}, "
                    f"{param.mean().item()}, "
                    f"{param.max().item()}, "
                )
            else:
                logging.info(f"{name}, {param.shape}, {param.device}, {param.dtype}, {param.requires_grad}")

def load_model_and_tokenizer(args):
    logging.info("Loading gold model ...")
    gold_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.gold_model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_auth_token=args.token,
        device_map=args.gold_model_device,
    )
    gold_model.eval()

    logging.info("Loading lora model ...")
    target_modules = ["query_proj", "key_proj", "value_proj", "output.dense", "intermediate.dense"]
    lora_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.lora_model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_auth_token=args.token,
        device_map=args.lora_model_device,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    lora_model = get_peft_model(lora_model, lora_config)
    lora_model.eval()

    logging.info("Loading tokenizer ...")
    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": False,
        "revision": "main",
        "trust_remote_code": False,
        "token": args.token,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.gold_model_name_or_path, **tokenizer_kwargs)
    return gold_model, lora_model, tokenizer


def get_dataloader(tokenizer, nsamples, seqlen, seed, dataset_name):
    logging.info(f"Tokenize {dataset_name} ...")
    if dataset_name == "pile":
        traindata = load_dataset("monology/pile-uncopyrighted", data_files={"train": "val.jsonl.zst"})
        trainenc = tokenizer("\n\n".join(traindata["train"]['text'][:(10 * nsamples)]), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    elif dataset_name == "wikitext":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    logging.info(f"Finished tokenization")
    return trainloader

@torch.no_grad()
def obtain_gold_output(gold_model, gold_model_device, input_ids, quantized_module):
    act_dicts = OrderedDict()
    def save_act_hook(m, x, y, name):
        act_dicts[name] = y.to("cpu")

    hooks = []
    for name, m in gold_model.named_modules():
        if isinstance(m, nn.Linear) and (name == quantized_module):
            hooks.append(m.register_forward_hook(functools.partial(save_act_hook, name=name)))

    gold_model(input_ids.to(device=gold_model_device))
    for hook in hooks:
        hook.remove()
    return act_dicts

@torch.no_grad()
def obtain_lora_input(lora_model, lora_model_device, input_ids, quantized_module):
    act_dicts = OrderedDict()
    def save_act_hook(m, x, y, name):
        act_dicts[name] = x[0].to("cpu")

    hooks = []
    for name, m in lora_model.named_modules():
        if isinstance(m, tuners.lora.Linear) and (name == quantized_module):
            hooks.append(m.register_forward_hook(functools.partial(save_act_hook, name=name)))

    lora_model(input_ids.to(device=lora_model_device))
    for hook in hooks:
        hook.remove()
    return act_dicts


def initialize_lora(
    args,
    config,
    gold_model,
    lora_model,
    dataloader,
    module,
    quantizer=None,
):
    if "intermediate.dense" in module:
        gold_outputs = torch.zeros(
            (args.num_samples, args.max_length, config.intermediate_size), dtype=torch.bfloat16, device="cpu")
        lora_inputs = torch.zeros(
            (args.num_samples, args.max_length, config.hidden_size), dtype=torch.bfloat16, device="cpu")
    elif ("output.dense" in module) and ("attention.output.dense" not in module):
        gold_outputs = torch.zeros(
            (args.num_samples, args.max_length, config.hidden_size), dtype=torch.bfloat16, device="cpu")
        lora_inputs = torch.zeros(
            (args.num_samples, args.max_length, config.intermediate_size), dtype=torch.bfloat16, device="cpu")
    else:
        gold_outputs = torch.zeros(
            (args.num_samples, args.max_length, config.hidden_size), dtype=torch.bfloat16, device="cpu")
        lora_inputs = torch.zeros(
            (args.num_samples, args.max_length, config.hidden_size), dtype=torch.bfloat16, device="cpu")

    lora_module = "base_model.model." + module
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            _output = obtain_gold_output(gold_model, args.gold_model_device, batch[0], module)
            gold_outputs[i] = _output[module]
            _input = obtain_lora_input(lora_model, args.lora_model_device, batch[0], lora_module)
            lora_inputs[i] = _input[lora_module]

    weight_quant_params = {
        "n_bits": args.bits,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": args.block_size,
        "lwc": True
    }

    for n, m in lora_model.named_modules():
        if n == lora_module:
            ori_lora_layer = m
            lora_layer = QuantLinear(ori_lora_layer, weight_quant_params)
    for n, m in gold_model.named_modules():
        if n == module:
            gold_layer = m

    weight = gold_layer.weight.clone()
    if args.bits in [2, 4, 8]:
        if quantizer is None:
            q_weight = bnb.nn.Params4bit(
                weight.to("cpu"),
                requires_grad=False,
                compress_statistics=False,
                quant_type="nf4"
            ).to("cuda")
            deq_weight = bnb.functional.dequantize_4bit(
                q_weight.data,
                q_weight.quant_state,
                quant_type="nf4"
            )
        else:
            q_weight, max_abs, shape = quantizer.quantize_block(weight)
            deq_weight = quantizer.dequantize_block(q_weight, max_abs, shape)
    else:
        deq_weight = None

    logging.info(f"=============={module}==============")
    with torch.no_grad():
        lora_layer.float().cuda()
    logging.info("Trainable parameters:")
    for n, p in lora_layer.named_parameters():
        if p.requires_grad:
            logging.info(n)

    optimizer = torch.optim.AdamW(lora_layer.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_func = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        loss_list = []
        for j in range(args.num_samples // args.batch_size):
            optimizer.zero_grad()

            index = j * args.batch_size
            lora_out = lora_layer(lora_inputs[index:index + args.batch_size, ].to("cuda").float())
            loss = loss_func(gold_outputs[index:index + args.batch_size, ].to("cuda").float(), lora_out)
            if not math.isfinite(loss.item()):
                logging.info("Loss is NAN, stopping training")
                continue

            loss_list.append(loss.detach().cpu())
            loss.backward()
            optimizer.step()

        if args.bits in [2, 4, 8]:
            logging.info(f"Epoch {epoch}: {torch.stack(loss_list).mean()} \t"
                         f"{torch.norm(weight - deq_weight)} vs "
                         f"{torch.norm(weight - lora_layer.weight_quantizer(ori_lora_layer.weight) - lora_layer.scaling * lora_layer.lora_B_weight @ lora_layer.lora_A_weight)}")
        else:
            logging.info(f"Epoch {epoch}: {torch.stack(loss_list).mean()} \t"
                         f"{torch.norm(weight - lora_layer.weight_quantizer(ori_lora_layer.weight) - lora_layer.scaling * lora_layer.lora_B_weight @ lora_layer.lora_A_weight)}")

    ori_lora_layer.weight.data = lora_layer.weight_quantizer(ori_lora_layer.weight.clone()).to(dtype=torch.bfloat16)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument(
        "--gold_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model",
    )
    parser.add_argument(
        "--lora_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub",
    )
    parser.add_argument(
        "--gold_model_device",
        type=str,
        default="cuda",
        help="The device for the gold(original) model",
    )
    parser.add_argument(
        "--lora_model_device",
        type=str,
        default="cuda",
        help="The device for the lora model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="The size of the calibration sentences",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="The max length of the tokenized sentence",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha of the LoRA adapter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.,
        help="The dropout of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="The directory for saving the newly initialized lora model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--custom_quantizer",
        action='store_true',
        help="Use custom quantizer instead of bitsandbytes"
    )
    parser.add_argument(
        "--quantized_method",
        choices=['normal', 'uniform'],
        type=str,
        help="Distribution of the quantized weight"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Block size for the quantization"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0,
        help="Weight decay"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--init_from_loftq",
        action='store_true',
    )
    parser.add_argument(
        "--cal_dataset_name",
        type=str,
        default="pile",
    )
    args = parser.parse_args()
    return args


def main(args):
    gold_model, lora_model, tokenizer = load_model_and_tokenizer(args)
    logging.info("Lora model:")
    logging.info(lora_model)

    dataloader = get_dataloader(tokenizer, args.num_samples, args.max_length, args.seed, args.cal_dataset_name)

    ordered_modules = [
        "deberta.encoder.layer.0.attention.self.query_proj", "deberta.encoder.layer.0.attention.self.key_proj",
        "deberta.encoder.layer.0.attention.self.value_proj", "deberta.encoder.layer.0.attention.output.dense",
        "deberta.encoder.layer.0.intermediate.dense", "deberta.encoder.layer.0.output.dense"]
    ordered_init_modules = []
    for l in range(gold_model.config.num_hidden_layers):
        tmp = []
        for module in ordered_modules:
            tmp.append(module.replace(".0.", f".{l}."))
        ordered_init_modules += tmp
    logging.info("Initialized orders:")
    logging.info(ordered_init_modules)

    quantizer = None
    if args.custom_quantizer and args.bits in [2, 4, 8]:
        quantizer = NFQuantizer(
            num_bits=args.bits,
            device="cuda",
            method=args.quantized_method,
            block_size=args.block_size
        )

    for module in ordered_init_modules:
        initialize_lora(
            args,
            gold_model.config,
            gold_model,
            lora_model,
            dataloader,
            module,
            quantizer=quantizer,
        )

    # Save
    base_model = lora_model.get_base_model()

    # Save Quantized model
    model_name = args.gold_model_name_or_path.split("/")[-1] + \
                 f"-{args.bits}bit" + \
                 f"-{args.lora_rank}rank" + \
                 f"-{args.lora_dropout}dropout"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "loftq_init")

    # save lora adapters first
    lora_model.base_model.peft_config[
        "default"
    ].base_model_name_or_path = base_model_dir  # This can be a local path or Hub model id
    lora_model.base_model.peft_config["default"].init_lora_weights = True  # Don't apply LoftQ when loading again

    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)
    print_model(base_model, "base_model")


if __name__ == "__main__":
    args = arg_parse()
    logging.info(args)
    main(args)