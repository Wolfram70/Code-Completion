# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script is used to generate the embedding vectors for the given dataset.

import argparse
import logging
import os
import random
import re
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, AutoTokenizer)

logger = logging.getLogger(__name__)

class InferDataset(Dataset):
    def __init__(self, tokenizer, args, api=True):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()
        self.tokenizer = tokenizer
        self.args = args
        self.api = api
        data_file = args.data_path

        from process_python import processor
        self.proc = processor(args.lang, remove_comments=False)

        logger.info(f"Creating features from {data_file}")
        data_format = data_file.split(".")[-1]

        self.data = []
        self.idx = []
        n = 0
        with open(data_file) as f:
            for _ in f:
                n += 1
        # n = 100000
        st = n//world_size*local_rank
        ed = n//world_size*(local_rank+1)
        logger.warning(f"device {local_rank} will load {ed-st} data line from {st} to {ed}")
        with open(data_file) as f:
            for i,line in enumerate(f):
                if i >= st and i < ed:
                    if (i-st) % 100000 == 0:
                        logger.info(f"device {local_rank} created {i-st}/{ed-st} train data")
                    if "json" in data_format:
                        content = json.loads(line)
                        self.data.append(self.convert_cxg_format_to_normal(content["input"]))
                        self.idx.append(content["id"])
                    else:   # txt
                        self.data.append(self.convert_cxg_format_to_normal(line.strip()))
                        self.idx.append(i)
        logger.warning(f"device {local_rank} loaded {len(self.data)} train data from {st} to {ed}")

    def convert_cxg_format_to_normal(self, code):
        if code.startswith("<s>"):
            code = code.lstrip("<s>")
        if code.endswith("</s>"):
            code = code.rstrip("</s>")
        code = code.replace("<EOL>", "\n")
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        lines = code.split("\n")
        indent_size = 4
        indent = 0
        res = ""
        for line in lines:
            indent += line.count("<INDENT>")
            indent -= line.count("<DEDENT>")
            res += "\n" + " "*indent_size*indent + line.replace("<INDENT>", "").replace("<DEDENT>", "")
        return res

    def encode(self, code, api_seq):
        if self.api:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + \
                          [self.tokenizer.sep_token] + self.tokenizer.tokenize(" ".join(api_seq)) + [self.tokenizer.sep_token]
        else:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + [self.tokenizer.sep_token]
        code_tokens = code_tokens[:self.args.block_size]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return code_ids
    
    def encode_print(self, code, api_seq):
        if self.api:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + \
                          [self.tokenizer.sep_token] + self.tokenizer.tokenize(" ".join(api_seq)) + [self.tokenizer.sep_token]
        else:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + [self.tokenizer.sep_token]
        code_tokens = code_tokens[:self.args.block_size]
        return code_tokens

    def process(self, code):
        self.proc.update(code)
        api_seq = self.proc.get_api_seq()
        code = self.proc.untokenize(cut_ratio=0.0)
        token_id = self.encode(code, api_seq)
        return token_id
    
    def process_print(self):
        for code in self.data:
            self.proc.update(code)
            api_seq = self.proc.get_api_seq()
            code = self.proc.untokenize(cut_ratio=0.0)
            code = self.proc.convert_to_normal(code)
            #tokens = self.encode_print(code, api_seq)
            print(code)
    
    def process_print_posex(self):
        for code in self.data:
            self.proc.update(code)
            api_seq = self.proc.get_api_seq()
            code = self.proc.process_train(cut_ratio=0.0)
            #tokens = self.encode_print(code, api_seq)
            print(code)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return [torch.tensor(self.process(self.data[item])), torch.tensor([self.idx[item]])] #torch.tensor(self.process_posex(self.data[item])), torch.tensor([self.idx[item]])]

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Language of the dataset.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/reacc-py-retriever")

    infer = InferDataset(tokenizer, args)

    infer.process_print()
    infer.process_print_posex()

if __name__ == "__main__":
    main()

