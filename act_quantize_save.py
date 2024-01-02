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
import logging
import argparse
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, tuners
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

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
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    logger.info("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    logger.info("=" * 10 + name + "=" * 10)
    logger.info(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                logger.info(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                logger.info(name, param.shape, param.device, param.dtype, param.requires_grad)

def load_model_and_tokenizer(args):
    logger.info("Loading gold model ...")
    gold_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_auth_token=args.token,
        device_map=args.gold_model_device,
    )
    gold_model.eval()

    logger.info("Loading lora model ...")
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    lora_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_auth_token=args.token,
        device_map=args.lora_model_device,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    lora_model = get_peft_model(lora_model, lora_config)
    lora_model.eval()

    logger.info("Loading tokenizer ...")
    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "trust_remote_code": False,
        "token": args.token,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    return gold_model, lora_model, tokenizer


def obtain_gold_activation(gold_model, gold_model_device, input_ids, quantized_modules):
    act_dicts = OrderedDict()

    def save_act_hook(m, x, y, name):
        if isinstance(x, tuple):
            act_dicts[name + '.input'] = x[0].squeeze().to("cpu")
        else:
            act_dicts[name + '.input'] = x.squeeze().to("cpu")
        act_dicts[name + '.output'] = y.squeeze().to("cpu")

    hooks = []
    for name, m in gold_model.named_modules():
        if isinstance(m, nn.Linear) and (name in quantized_modules):
            hooks.append(m.register_forward_hook(functools.partial(save_act_hook, name=name)))

    gold_model(input_ids.to(gold_model_device))
    for hook in hooks:
        hook.remove()
    return act_dicts

def low_rank_decomposition(weight, lora_rank=32):
    assert len(weight.size()) == 2, "Only Support 2D matrix"
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    L = U @ (torch.sqrt(torch.diag(S)[:, 0:lora_rank]))
    R = torch.sqrt(torch.diag(S)[0:lora_rank, :]) @ Vh
    return L, R

@torch.no_grad()
def initialize_lora(
    gold_model,
    lora_model,
    tokenizer,
    gold_model_device,
    lora_model_device,
    dataset,
    quantized_modules,
    lora_rank,
    num_samples=128,
    max_length=256,
    threshold_scale=5,
):
    lora_As = OrderedDict()
    lora_Bs = OrderedDict()
    quantized_weights = OrderedDict()
    lora_quantized_modules = []
    for module in quantized_modules:
        lora_quantized_modules.append("base_model.model." + module)

    for i in range(num_samples):
        input_ids = tokenizer(
            dataset["train"][i]["text"],
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).input_ids

        # obtain gold activation
        gold_act_dicts = obtain_gold_activation(gold_model, gold_model_device, input_ids, quantized_modules)

        ## initialize lora_A and lora_B
        def lora_init_hook(m, x, y, name):
            dtype = y.dtype
            gold_name = name[len("base_model.model."):]
            gold_x = gold_act_dicts[gold_name + ".input"].to("cuda", dtype=torch.float32)
            gold_y = gold_act_dicts[gold_name + ".output"].to("cuda", dtype=torch.float32)
            lora_x = x[0].clone().to("cuda", dtype=torch.float32).squeeze()

            weight = m.weight.clone()
            weight = weight.to(device="cuda", dtype=torch.float32)
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
            ).to(dtype=torch.float32)

            res = (gold_y - lora_x @ deq_weight.T) / m.scaling["default"]
            L, R = low_rank_decomposition(torch.linalg.lstsq(lora_x, res).solution.T, lora_rank=lora_rank)

            ori_err = torch.norm(gold_y - gold_x @ deq_weight.T)
            new_err = torch.norm(gold_y - lora_x @ deq_weight.T - m.scaling['default'] * lora_x @ torch.mm(L, R).T)
            ori_weight_err = torch.norm(weight - deq_weight)
            weight_err = torch.norm(weight - deq_weight - m.scaling['default'] * torch.mm(L, R))

            if (not torch.isnan(deq_weight).any()) and (not torch.isnan(R).any()) and (not torch.isnan(L).any()) \
                    and new_err < threshold_scale * ori_err and weight_err < threshold_scale * ori_weight_err:
                if name in lora_As:
                    lora_As[name] = lora_As[name] + R.to(device=lora_model_device, dtype=dtype)
                    lora_As[name + ".count"] += 1
                else:
                    lora_As[name] = R.to(device=lora_model_device, dtype=dtype)
                    lora_As[name + ".count"] = 1

                if name in lora_Bs:
                    lora_Bs[name] = lora_Bs[name] + L.to(device=lora_model_device, dtype=dtype)
                    lora_Bs[name + ".count"] += 1
                else:
                    lora_Bs[name] = L.to(device=lora_model_device, dtype=dtype)
                    lora_Bs[name + ".count"] = 1

                quantized_weights[name] = deq_weight.to(device=lora_model_device, dtype=dtype)

        hooks = []
        for name, m in lora_model.named_modules():
            if isinstance(m, tuners.lora.Linear) and (name in lora_quantized_modules):
                hooks.append(m.register_forward_hook(functools.partial(lora_init_hook, name=name)))

        lora_model(input_ids.to(lora_model_device))
        for hook in hooks:
            hook.remove()

    ## aggregate lora_A and lora_B's weights
    for name, m in lora_model.named_modules():
        if isinstance(m, tuners.lora.Linear) and (name in lora_quantized_modules):
            m.weight.data = quantized_weights[name]
            if name in lora_As:
                logger.info(f"Initialize lora_A of {name} ...")
                m.lora_A["default"].weight.data = lora_As[name] / lora_As[name + ".count"]
            else:
                logger.info(f"lora_A of {name} stays unchanged!")
            if name in lora_Bs:
                logger.info(f"Initialize lora_B of {name} ...")
                m.lora_B["default"].weight.data = lora_Bs[name] / lora_Bs[name + ".count"]
            else:
                logger.info(f"lora_B of {name} stays unchanged!")


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument(
        "--model_name_or_path",
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
    args = parser.parse_args()
    return args

def main(args):
    gold_model, lora_model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        data_files={
            "train": "val.jsonl.zst",
        },
    )
    dataset = dataset.shuffle(seed=args.seed)

    ordered_moduels = [
        ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj"],
        ["model.layers.0.self_attn.o_proj"],
        ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
        ["model.layers.0.mlp.down_proj"],
    ]
    ordered_init_modules = []
    for l in range(gold_model.config.num_hidden_layers):
        for block_modules in ordered_moduels:
            temp = []
            for module in block_modules:
                temp.append(module.replace(".0.", f".{l}."))
            ordered_init_modules.append(temp)
    logger.info(f"Ordered init modules: {ordered_init_modules}")

    for modules in ordered_init_modules:
        initialize_lora(
            gold_model,
            lora_model,
            tokenizer,
            args.gold_model_device,
            args.lora_model_device,
            dataset,
            modules,
            args.lora_rank,
            num_samples=args.num_samples,
            max_length=args.max_length
        )

    # Save
    base_model = lora_model.get_base_model()

    # Save Quantized model
    model_name = args.model_name_or_path.split("/")[-1] + \
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
    main(args)