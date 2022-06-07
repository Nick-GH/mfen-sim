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
import sys

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.utils.data.sampler import BatchSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FuncDatasetAllInput(Dataset):
    def __init__(self, file_name, split, pretrain_tokenizer, codebert_tokenizer, limit, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.max_length = opt.max_length

        self.min_size = opt.min_size
        self.max_size = opt.max_size
        self.limit = limit

        self.max_length_type = opt.max_length_type

        self.pretrain_tokenizer = pretrain_tokenizer
        self.codebert_tokenizer = codebert_tokenizer
        

        self.func_datas = []
        self.ground_truth2func_ids = {}
        self.ground_truth2index = {}

        self.read_cache = opt.read_cache
        self.cache_path = opt.cache_path

        if self.read_cache and os.path.exists(self.cache_path + self.split + "_func_datas.pickle"):
            print("Reading cache...")
            with open(self.cache_path + self.split + "_func_datas.pickle", "rb") as f:
                self.func_datas = pickle.load(f)
            with open(self.cache_path + self.split + "_ground_truth_map.pickle", "rb") as f:
                self.ground_truth2func_ids = pickle.load(f)
            print("Cache read.")
        else:
            self._preprocess()
        
        ground_truths = list( self.ground_truth2func_ids.keys() )
        for i,ground_truth in enumerate(ground_truths):
            self.ground_truth2index[ground_truth] = i
        self.labels = []
        for func_data in self.func_datas:
            ground_truth = func_data.ground_truth
            self.labels.append(self.ground_truth2index[ground_truth])
        self.labels = np.array(self.labels)
    
    def truncate_graph(self, func_datas):
        
        filtered_func_datas = []
        count = 0
        for func_data in func_datas:
            g = nx.DiGraph()
            cfg_size = func_data["cfg_size"]
            if cfg_size >= self.min_size and cfg_size <= self.max_size:
                filtered_func_datas.append(func_data)
                continue
            if cfg_size < self.min_size:
                continue
            cfg_edges = func_data["cfg"]
            g.add_edges_from(cfg_edges)
            func_data["bb_data"] = func_data["bb_data"][: self.max_size]
            remove_nodes = [i for i in range(self.max_size, cfg_size)]
            g.remove_nodes_from(remove_nodes)
            func_data["cfg"] = list(g.edges)
            func_data["cfg_size"] = len(func_data["bb_data"])
            count += 1
            filtered_func_datas.append(func_data)
        return filtered_func_datas
        
    def _preprocess(self):
        print("Loading func datas from {}...".format(self.data_dir) )
        # print(self.data_dir)
        assert os.path.exists(self.data_dir)

        with open(self.data_dir, "rb") as f:

            func_datas = pickle.load(f)
            func_datas = self.truncate_graph(func_datas)
            func_datas = func_datas[:self.limit]
            print("Total loading {} functions".format(str(len(func_datas))) )
            print("Func data loaded.")
            for i,func_data in tqdm(enumerate(func_datas)):
                function = Function(func_data)
                ground_truth = function.ground_truth

                self.func_datas.append(function)
                index = len(self.func_datas)-1
                if ground_truth not in self.ground_truth2func_ids:
                    self.ground_truth2func_ids[ground_truth] = [index]
                else:
                    self.ground_truth2func_ids[ground_truth].append(index)

        print("Saving cache...")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(self.cache_path + self.split + "_func_datas.pickle", "wb") as f:
            pickle.dump(self.func_datas, f)
        with open(self.cache_path + self.split + "_ground_truth_map.pickle", "wb") as f:
            pickle.dump(self.ground_truth2func_ids, f)
        print("Cache saved.")



    def __len__(self):
        return len(self.func_datas)
   
    def __getitem__(self, index):
        function = self.func_datas[index]
        bb_insts = function.get_bb_insts()
        edges = function.get_cfg_edge_list()
        ground_truth = function.ground_truth
        func_strings_and_consts = self.get_func_strings_and_consts(function, self.codebert_tokenizer.sep_token)

        graph, type_input, sc_input = self.get_input_ids(bb_insts, edges, func_strings_and_consts)
        
        label = self.ground_truth2index[ground_truth]
        arch = function.arch
        compiler = function.compiler
        opti = function.opti
     
   
        ground_truth = function.ground_truth
        
        output = {
            "index": int(index),
            "graph": graph,
            "type_input": type_input, # should be fed into model
            "sc_input": sc_input, # should be fed into model
            "arch": arch,
            "complier": compiler,
            "opti": opti,
            "label": label, # label for online selector, is a tuple containing strings
            
        }
        return output
    
    def get_input_ids(self, func_bb_insts, edges, func_strings_and_consts):
        func_bb_inst_ids = self.pretrain_tokenizer(func_bb_insts, padding="max_length", max_length = self.max_length, truncation=True)
        func_bb_inst_ids["input_ids"] = torch.Tensor(func_bb_inst_ids["input_ids"]).long()
        func_bb_inst_ids["token_type_ids"] = torch.Tensor(func_bb_inst_ids["token_type_ids"]).long()
        func_bb_inst_ids["attention_mask"] = torch.Tensor(func_bb_inst_ids["attention_mask"]).long()
        

        func_insts = " ".join(func_bb_insts).strip()
        func_inst_ids = self.pretrain_tokenizer(func_insts, padding="max_length", max_length = self.max_length_type, truncation=True)

        func_inst_ids["input_ids"] = torch.Tensor(func_inst_ids["input_ids"]).long()
        func_inst_ids["token_type_ids"] = torch.Tensor(func_inst_ids["token_type_ids"]).long()
        func_inst_ids["attention_mask"] = torch.Tensor(func_inst_ids["attention_mask"]).long()


        edges = torch.Tensor(edges).long()

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

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, limit = -1, n_classes=10, n_samples=2):
        self.labels = labels
        self.labels_set = set(labels)
        self.label2indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        # 过滤indices少于2的label
        for l in self.labels_set:
            if len(self.label2indices[l]) < 2:
                del self.label2indices[l]
        self.labels_set = self.label2indices.keys()
        print("Total  {} ground truths.".format(len(self.labels_set)))
        for l in self.labels_set:
            np.random.shuffle(self.label2indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        if limit == -1:
            self.n_dataset = len(self.labels)
        else:
            self.n_dataset = min(limit, len(self.labels))
        self.batch_size = self.n_samples * self.n_classes

        self.labels_list = list(self.labels_set)

    def __iter__(self):
        self.count = 0
        while self.count < self.n_dataset:
            self.index = 0
            random.shuffle(self.labels_list)
            while self.index < len(self.labels_list) and self.count < self.n_dataset:
                classes = self.labels_list[self.index: self.index + self.n_classes]
                indices = []
                for class_ in classes:
                    indices.extend(self.label2indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                    self.used_label_indices_count[class_] += self.n_samples
                    if self.used_label_indices_count[class_] + self.n_samples > len(self.label2indices[class_]):
                        np.random.shuffle(self.label2indices[class_])
                        self.used_label_indices_count[class_] = 0
                yield indices
                self.index += self.n_classes 
                self.count += self.batch_size 

    def __len__(self):
        return self.n_dataset // self.batch_size



def get_dataset(data_file, split, tokenizer, limit, opt):
    return FuncDatasetAllInput(data_file, split, tokenizer,  limit, opt)


def get_data_loader(data_file, split,  tokenizer, codebert_tokenizer, limit, opt, batch_size, shuffle=True,  num_workers=0, collate_fn = None):
    dataset = FuncDatasetAllInput(data_file, split, tokenizer, codebert_tokenizer, limit, opt)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None)
    return loader

def get_batch_sample_data_loader(data_file, split,  tokenizer, codebert_tokenizer, limit, opt, n_classes, n_samples, num_workers=0, collate_fn = None):
    dataset = FuncDatasetAllInput(data_file, split,tokenizer, codebert_tokenizer, -1, opt)
    batch_sampler = BalancedBatchSampler(dataset.labels, limit=limit, n_classes=n_classes, n_samples=n_samples)
    loader = DataLoader(dataset=dataset,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None,
                             batch_sampler=batch_sampler)
    return loader

if __name__=="__main__":
    pass