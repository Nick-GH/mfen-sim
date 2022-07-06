# coding:utf-8
import enum
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
from torch.utils.data import Dataset
from transformers.file_utils import logger, logging
from transformers import BertTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import  BertWordPieceTokenizer
from .data import Function, BasicBlock
import pandas as pd
import copy
import gc
from sklearn.model_selection import train_test_split
from collections import defaultdict
import networkx as nx
import time
import sys

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
import collections




def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def get_data_loader(opt, func_datas, batch_size, num_workers=4):
    dataset = FuncEncoder(opt, func_datas)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None)
    return loader

class FuncEncoder(Dataset):

    def __init__(self,  opt, func_datas) -> None:
        self.opt = opt
 
        self.max_length = opt.max_length

        self.min_size = opt.min_size
        self.max_size = opt.max_size 

        self.max_length_type = opt.max_length_type

        vocab_file = opt.vocab_file

        vocab = load_vocab(vocab_file=vocab_file)
        pretrain_tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=True, never_split=vocab)
        codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.pretrain_tokenizer = pretrain_tokenizer
        self.codebert_tokenizer = codebert_tokenizer

        self.func_datas = func_datas
    
    def __len__(self):
        return len(self.func_datas)
    
    def __getitem__(self, index):
        return self.encode(self.func_datas[index])
    
    def truncate_graph(self, func_data):
        if isinstance(func_data, Dict):
            g = nx.DiGraph()
            cfg_size = func_data["cfg_size"]
            if cfg_size >= self.min_size and cfg_size <= self.max_size:
                return 
            if cfg_size < self.min_size:
                return

            cfg_edges = func_data["cfg"]
            g.add_edges_from(cfg_edges)
            func_data["bb_data"] = func_data["bb_data"][: self.max_size]
            remove_nodes = [i for i in range(self.max_size, cfg_size)]
            g.remove_nodes_from(remove_nodes)
            func_data["cfg"] = list(g.edges)
            func_data["cfg_size"] = len(func_data["bb_data"])
        elif isinstance(func_data, Function):
            g = nx.DiGraph()
            cfg_size = func_data.cfg_size
            if cfg_size >= self.min_size and cfg_size <= self.max_size:
                return 
            if cfg_size < self.min_size:
                return
            cfg_edges = func_data.cfg_edge_list
            g.add_edges_from(cfg_edges)
            func_data.bb_data = func_data.bb_data[: self.max_size]
            remove_nodes = [i for i in range(self.max_size, cfg_size)]
            g.remove_nodes_from(remove_nodes)
            func_data.cfg_edge_list = list(g.edges)
            func_data.cfg_size = len(func_data.cfg_edge_list)
    
    
    def encode(self, function):
        # function: list[bb_inst],list[bb string], list[bb const], edges [srouce_list, target_list ]
        # type: func_inst
        # bb_insts, edges, bb_info, type_info,  ground_truth
        self.truncate_graph(function)
        if type(function) is dict:
            function = Function(function)
        func_bb_insts = function.get_bb_insts()
        edges = function.get_cfg_edge_list()
        func_strings_and_consts = self.get_func_strings_and_consts(function, self.codebert_tokenizer.sep_token)

        graph, type_input, sc_input = self.get_input_ids(func_bb_insts, edges, func_strings_and_consts)
        
        output = {
            "graph": graph, # edges, func_bb_insts, sc_embedding
            "type_input": type_input, 
            "sc_input": sc_input,
        }
        return output
    

    def get_input_ids(self, func_bb_insts, edges, func_strings_and_consts):
        func_bb_inst_ids = self.pretrain_tokenizer(func_bb_insts, padding="max_length", max_length = self.max_length, truncation=True)
        func_bb_inst_ids["input_ids"] = torch.Tensor(func_bb_inst_ids["input_ids"]).long()
        func_bb_inst_ids["token_type_ids"] = torch.Tensor(func_bb_inst_ids["token_type_ids"]).long()
        func_bb_inst_ids["attention_mask"] = torch.Tensor(func_bb_inst_ids["attention_mask"]).long()
        

        func_insts = " ".join(func_bb_insts).strip()
        func_inst_ids = self.pretrain_tokenizer(func_insts, padding="max_length", max_length = self.max_length_type, truncation=True)
        # print(func_insts)
        # print(self.pretrain_tokenizer.decode(func_inst_ids["input_ids"]))
        func_inst_ids["input_ids"] = torch.Tensor(func_inst_ids["input_ids"]).long()
        func_inst_ids["token_type_ids"] = torch.Tensor(func_inst_ids["token_type_ids"]).long()
        func_inst_ids["attention_mask"] = torch.Tensor(func_inst_ids["attention_mask"]).long()
        # print(self.pretrain_tokenizer.decode(func_inst_ids["input_ids"]))

        # edges = function.get_cfg_edge_list() # List[ List, List ]
        edges = torch.Tensor(edges).long()

        # func_bb_strings_and_consts = self.get_func_bb_strings_and_consts(function, self.codebert_tokenizer.cls_token, self.codebert_tokenizer.sep_token)
        func_strings_and_consts_ids = self.codebert_tokenizer(func_strings_and_consts, padding="max_length", max_length = self.max_length, truncation=True) # bb_num * seq_len
        func_strings_and_consts_ids["input_ids"] = torch.Tensor(func_strings_and_consts_ids["input_ids"]).long()
        func_strings_and_consts_ids["attention_mask"] = torch.Tensor(func_strings_and_consts_ids["attention_mask"]).long()


        graph = Data(edge_index=edges, num_nodes=len(func_bb_insts), func_bb_inst_ids = func_bb_inst_ids)

        return graph, func_inst_ids, func_strings_and_consts_ids
    
    
    def get_func_strings_and_consts(self, function,  sep_token, const_first=False):
        strings = function.get_func_strings()
        consts = function.get_func_consts()
        if not const_first:
            strings_and_consts = " ".join([strings, sep_token, consts])
        else:
            strings_and_consts = " ".join([consts, sep_token, strings])
        
        return strings_and_consts

if __name__=="__main__":
    pass