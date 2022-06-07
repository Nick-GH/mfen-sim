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
from torch_geometric.loader import DataLoader #
from torch.utils.data.sampler import BatchSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FuncDataset(Dataset):
    def __init__(self, file_name, split, pretrain_tokenizer,  limit, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.max_length = opt.max_length

        self.limit = limit

        self.max_length_type = opt.max_length_type
        self.arch_same = opt.arch_same
        self.compiler_same = opt.compiler_same
        self.opti_same = opt.opti_same

        self.pretrain_tokenizer = pretrain_tokenizer

        self.func_datas = []
        self.ground_truth2func_ids = {}
        self.ground_truth2index = {}

        self.read_cache = opt.read_cache
        self.cache_path = opt.cache_path
        self.embedding_path = opt.embedding_path

        if self.read_cache and os.path.exists(self.cache_path + self.split + "_func_features.pickle"):
            print("Reading cache...")
            with open(self.cache_path + self.split + "_func_features.pickle", "rb") as f:
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
            ground_truth = func_data[-1]
            self.labels.append(self.ground_truth2index[ground_truth])
        self.labels = np.array(self.labels) 
        print("Total loading {} functions.".format(len(self.func_datas)))
        
    def _preprocess(self):
        print("Loading func features from {}...".format(self.data_dir) )
        assert os.path.exists(self.data_dir)

        with open(self.data_dir, "rb") as f:
            print("Loading sp embeddings...")
            type_infos = torch.load(self.embedding_path + self.split + "_sp_embeddings.pt")
            type_infos = type_infos.numpy()
            print(type_infos)
            print(type_infos.shape)
            print("Loading cl embeddings...")
            sc_infos = torch.load(self.embedding_path  + self.split + "_cl_embeddings.pt")
            print(sc_infos)
            sc_infos = sc_infos.numpy()
            print(sc_infos.shape)

            func_datas = pickle.load(f)
            print(len(func_datas))
            func_datas = func_datas[:self.limit]

            print("Total loading {} functions".format(str(len(func_datas))) )
            print("Func data loaded.")
            for i,func_data in tqdm(enumerate(func_datas)):
                function = Function(func_data)
                bb_insts = function.get_bb_insts()
                edges = function.get_cfg_edge_list()
                ground_truth = function.ground_truth
                sc_info = sc_infos[i]
                type_info = type_infos[i]
                self.func_datas.append((bb_insts, edges, type_info, sc_info, 
                    function.arch, function.compiler, function.opti, ground_truth))
                index = len(self.func_datas)-1
                if ground_truth not in self.ground_truth2func_ids:
                    self.ground_truth2func_ids[ground_truth] = [index]
                else:
                    self.ground_truth2func_ids[ground_truth].append(index)
        print("Total {} functions...".format(len(self.func_datas)))

        print("Saving cache...")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(self.cache_path + self.split + "_func_features.pickle", "wb") as f:
            pickle.dump(self.func_datas, f)
        with open(self.cache_path + self.split + "_ground_truth_map.pickle", "wb") as f:
            pickle.dump(self.ground_truth2func_ids, f)
        print("Cache saved.")



    def __len__(self):
        return len(self.func_datas)
   
    def __getitem__(self, index):
        bb_insts, edges, type_info, sc_info, arch, compiler, opti, ground_truth = self.func_datas[index]

        graph = self.get_input_graph(bb_insts, edges)
        
        type_info = torch.Tensor(type_info)
        sc_info = torch.Tensor(sc_info)
        label = self.ground_truth2index[ground_truth]
        
        output = {
            "index": int(index),
            "graph": graph,
            "type_input": type_info, 
            "sc_input": sc_info,
            "arch": arch,
            "complier": compiler,
            "opti": opti,
            "label": label, 
            
        }
        return output
    

    def get_input_graph(self, func_bb_insts, edges):
        
        func_bb_inst_ids = self.pretrain_tokenizer(func_bb_insts, padding="max_length", max_length = self.max_length, truncation=True)
        func_bb_inst_ids["input_ids"] = torch.Tensor(func_bb_inst_ids["input_ids"]).long() # bb_num * seq_len
        func_bb_inst_ids["token_type_ids"] = torch.Tensor(func_bb_inst_ids["token_type_ids"]).long()
        func_bb_inst_ids["attention_mask"] = torch.Tensor(func_bb_inst_ids["attention_mask"]).long()

        edges = torch.Tensor(edges).long() 
        graph = Data(edge_index=edges, num_nodes=len(func_bb_insts), func_bb_inst_ids = func_bb_inst_ids)

        return graph
    
    def get_func_bb_strings_and_consts(self, function, cls_token, sep_token, const_first=False):
        func_bb_strings_and_consts = []
        for bb in function.bb_data:
            strings = bb.get_strings()
            consts = bb.get_consts()
            if not const_first:
                strings_and_consts = " ".join([cls_token, strings, sep_token, consts, sep_token])
            else:
                strings_and_consts = " ".join([cls_token, consts, sep_token, strings, sep_token])
            func_bb_strings_and_consts.append(strings_and_consts)
        
        return func_bb_strings_and_consts


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
                    indices.extend(self.label2indices[class_][
                                self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                            class_] + self.n_samples])
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
    return FuncDataset(data_file, split, tokenizer,  limit, opt)


def get_data_loader(data_file, split,  tokenizer, limit, opt, batch_size, shuffle=True,  num_workers=0, collate_fn = None):
    dataset = FuncDataset(data_file, split,tokenizer, limit, opt)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None)
    return loader

def get_batch_sample_data_loader(data_file, split,  tokenizer, limit, opt, n_classes, n_samples, num_workers=0, collate_fn = None):
    dataset = FuncDataset(data_file, split,tokenizer, -1, opt)
    batch_sampler = BalancedBatchSampler(dataset.labels, limit=limit, n_classes=n_classes, n_samples=n_samples)
    loader = DataLoader(dataset=dataset,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None,
                             batch_sampler=batch_sampler)
    return loader

if __name__=="__main__":
    pass