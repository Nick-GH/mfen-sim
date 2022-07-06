# coding:utf-8


import os
import pickle
import random
import torch
import warnings
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers.file_utils import logger, logging
from transformers import BertTokenizer, PreTrainedTokenizerFast
from tokenizers import  BertWordPieceTokenizer
from .data import Function, BasicBlock
import pandas as pd
import copy
import gc
from sklearn.model_selection import train_test_split
from collections import defaultdict



# 该dataset加载train和valid
class BertClsDataset(Dataset):
    def __init__(self, file_name, split, tokenizer, limit, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.max_length = opt.max_length
        self.max_pos_length = opt.max_pos_length
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.limit = limit
        self.pred_arg_num = opt.pred_arg_num

        self.read_cache = opt.read_cache
        self.cache_path = opt.cache_path

        self.func_datas = []
        self.type_tags = ["int", "void", "char", "struct", "void*", "enum", "func", "short", "float", "union", "none"]
        self.type_tags2id = {}
        for i,type_tag in  enumerate(self.type_tags):
            self.type_tags2id[type_tag] = i
        
        if self.read_cache and os.path.exists(self.cache_path + self.split + "_cache.pickle"):
            print("Reading cache...")
            with open(self.cache_path + self.split + "_cache.pickle", "rb") as f:
                self.func_datas = pickle.load(f)
            print("{} cache read.".format(split))
            print("Load {} functions".format(len(self.func_datas)))
        else:
            self.read_function_datas()
        
        
    

    def read_function_datas(self):
        print("Loading func datas from {}...".format(self.data_dir))
        if os.path.isdir(self.data_dir):
            self.read_from_dir(self.data_dir)
        elif os.path.isfile(self.data_dir):
            if self.data_dir.endswith(".txt"):
                self.read_from_input_list(self.data_dir)
            else:
                self.read_from_file(self.data_dir)
        
        print("Total loading {} functions".format( str(len(self.func_datas))) )
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(self.cache_path + self.split + "_cache.pickle", "wb") as f:
            pickle.dump(self.func_datas, f)
            print("Cache saved.")
    
    def read_from_input_list(self,input_list):
        with open(input_list,"r") as f:
            binary_list = f.readlines()
            for file in binary_list:
                pickle_file = file.strip(os.linesep)+".pickle"
                self.read_from_file(pickle_file)
        pass

    def read_from_dir(self, data_dir):
        for root,dir,files in os.walk(data_dir):
            for file in files:
                if file.endswith("elf.pickle") or (file.endswith(".pickle") and self.split in file):
                    file_path = os.path.join(root,file)
                    self.read_from_file(file_path)

    def read_from_file(self, file):
        assert os.path.exists(file)
        with open(file, "rb") as f:
            func_datas = pickle.load(f)
            print("Func data loaded.")

            for func_data in tqdm(func_datas):
                function = Function(func_data)
                func_insts = function.get_bb_insts() # list[string]
                func_insts = " ".join(func_insts)
                labels = []
                abstract_args_type = function.abstract_args_type
                arg_num = len(abstract_args_type)
                if arg_num >= 10:
                    arg_num = 10
                labels.append(arg_num)
                for i in range(10):
                    if i < len(abstract_args_type):
                        arg_type = abstract_args_type[i]
                        arg_type = self.norm_abstract_type(arg_type)
                        labels.append(self.type_tags2id[arg_type])
                    else:
                        labels.append(self.type_tags2id["none"])
                self.func_datas.append((func_insts, labels))
            print("Total loading {} functions from file {}".format(str(len(func_datas)), file) )
        del func_datas
        gc.collect()

    def __len__(self):
        return len(self.func_datas)
    
    def __getitem__(self, index):
       
        func_insts, labels = self.func_datas[index]
      
        labels = labels[ : self.pred_arg_num + 1]
      
        func_inst_ids = self.tokenizer(func_insts, padding="max_length", max_length = self.max_length, truncation=True) # inst_num * seq_len
        output = {
            "input_ids": torch.Tensor(func_inst_ids["input_ids"]).long(), # bb_num * seq_len
            "token_type_ids": torch.Tensor(func_inst_ids["token_type_ids"]).long(), # bb_num * seq_len
            "attention_mask": torch.Tensor(func_inst_ids["attention_mask"]).long(), # bb_num * seq_len
            "labels": torch.Tensor(labels).long(), 
        }
        
        return output
    
    def norm_abstract_type(self, ret_type):
        ret_type = str(ret_type)
        if "*" in ret_type:
            # ret_type = "void*"
            ret_type = ret_type.split()[0]
            if ret_type == "void":
                ret_type = "void*"
        return ret_type

# map style dataset
def collate_fn(features) -> Dict[str, Tensor]:
    return

def get_dataset(data_file, split, tokenizer, limit, opt):
    return BertClsDataset(data_file, split, tokenizer, limit, opt)

def get_data_loader(data_file, split, tokenizer, limit, opt, batch_size, shuffle,  num_workers=4, collate_fn = None):
    dataset = BertClsDataset(data_file, split, tokenizer, limit, opt)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None)
    return loader

if __name__=="__main__":
    pass