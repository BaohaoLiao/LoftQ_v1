#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import PeftModel, get_peft_model, TaskType, LoraConfig


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    full_precision:  bool = field(default=False)
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )

@dataclass
class DataArguments:
    dataset_name: str = field(default="wikitext", metadata={"help": "Dataset name."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})


def evaluation(model_args, data_args):
    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            device_map='auto',
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            device_map='auto',
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
    ##########################
    #       Peft Model       #
    ##########################
    if model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=False,
            token=model_args.token,
        )
    model.config.use_cache = False
    model = model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, token=model_args.token, use_fast=False)

    if data_args.dataset_name == "wikitext":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testloader = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    else:
        raise ValueError("Please specify the dataset name.")

    # Evaluate
    seqlen = data_args.max_seq_length
    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs[0]
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        logging.info(f'{data_args.dataset_name} : {ppl.item()}')


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    evaluation(model_args, data_args)
