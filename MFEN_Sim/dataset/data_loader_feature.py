# coding:utf-8
import enum
import os
import pickle
import random
import torch
import warnings
from torch.autograd.grad_mode import F
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
import networkx as nx
import time
from transformers import BertTokenizer,  AutoTokenizer, BertModel



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureDataset(Dataset):
    def __init__(self, file_name, split, codebert_tokenizer, limit, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)

        self.max_length = opt.max_length

        self.limit = limit

        self.codebert_tokenizer = codebert_tokenizer

        self.func_sc = []

        self.read_cache = opt.read_cache
        self.cache_path = opt.cache_path

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
        
        print("Total loading {} functions".format( str(len(self.func_sc))) )
    
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
        print("Loading func datas from {}...".format(self.data_dir) )
        assert os.path.exists(self.data_dir)
        

        # 缓存设置
        with open(file, "rb") as f:
            func_datas = pickle.load(f)
            func_datas = func_datas[:self.limit]
            print("Total loading {} functions".format(str(len(func_datas))) )
            print("Func data loaded.")
            for func_data in tqdm(func_datas):
                function = Function(func_data)
                func_strings_and_consts = self.get_func_strings_and_consts(function, self.codebert_tokenizer.cls_token, self.codebert_tokenizer.sep_token)
                self.func_sc.append(func_strings_and_consts)
            
        
        del func_datas
        gc.collect()
    
    def __len__(self):
        return len(self.func_sc)
    

    def __getitem__(self, index):

        func_strings_and_consts = self.func_sc[index]
        func_strings_and_consts_ids = self.codebert_tokenizer(func_strings_and_consts, 
            padding="max_length", 
            max_length = self.max_length, 
            truncation=True) 
        func_strings_and_consts_ids["input_ids"] = torch.Tensor(func_strings_and_consts_ids["input_ids"]).long()
        func_strings_and_consts_ids["attention_mask"] = torch.Tensor(func_strings_and_consts_ids["attention_mask"]).long()

        return func_strings_and_consts_ids
    
    
    def get_func_bb_strings_and_consts(self, function, sep_token, const_first=False):
        func_bb_strings_and_consts = []
        for bb in function.bb_data:
            strings = bb.get_strings()
            consts = bb.get_consts()
            if not const_first:
                strings_and_consts = " ".join([strings, sep_token, consts])
            else:
                strings_and_consts = " ".join([consts, sep_token, strings])
            func_bb_strings_and_consts.append(strings_and_consts)
        
        return func_bb_strings_and_consts
    
    def get_func_strings_and_consts(self, function,  sep_token, const_first=False):
        strings = function.get_func_strings()
        consts = function.get_func_consts()
        if not const_first:
            strings_and_consts = " ".join([strings, sep_token, consts])
        else:
            strings_and_consts = " ".join([consts, sep_token, strings])
        
        return strings_and_consts


def collate_fn_bb(features) -> Dict[str, Tensor]:
    batch_bb_count = 0
    batch = [0]
    for i,feature in enumerate(features):
        # feature是dict
        bb_count = feature["input_ids"].shape[0]
        batch_bb_count += bb_count

        batch.append(batch_bb_count)
    
    output = {}
    
    elem = features[0]
    for key in elem.keys():
        all_features = features[0][key]
        for i in range(1,len(features)):
            all_features = torch.cat([all_features, features[i][key]], dim=0)
        output[key] = all_features
    output["batch"] = batch
    # print(output["input_ids"].shape)
    # print(output["attention_mask"].shape)
    # print(output["batch"])
    return output



def get_dataset(data_file, split, tokenizer, limit, opt):
    return FeatureDataset(data_file, split, tokenizer, limit, opt)


def get_data_loader(data_file, split,  tokenizer, limit, opt, batch_size, shuffle=False,  num_workers=0, collate_fn = None):
    dataset = FeatureDataset(data_file, split, tokenizer,  limit, opt)

    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader

if __name__=="__main__":
    pass