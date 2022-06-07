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





os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FuncPairDataset(Dataset):
    def __init__(self, file_name, split, pretrain_tokenizer,  limit, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.max_length = opt.max_length

        self.limit = limit

        self.max_length_type = opt.max_length_type
        self.neg_prob = opt.neg_prob 
        self.arch_same = opt.arch_same
        self.compiler_same = opt.compiler_same
        self.opti_same = opt.opti_same

        self.pretrain_tokenizer = pretrain_tokenizer

        self.func_datas = []
        self.ground_truth2func_ids = {}

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
            print(len(self.func_datas))
        else:
            self._preprocess()

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

                self.func_datas.append( (bb_insts, edges, type_info, sc_info, 
                    function.arch, function.compiler, function.opti, ground_truth) ) 
                
                
                
                if self.arch_same:
                    ground_truth += (function.arch,)
                if self.compiler_same:
                    ground_truth  += (function.compiler,)
                if self.opti_same:
                    ground_truth += (function.opti,)
                if ground_truth not in self.ground_truth2func_ids:
                    self.ground_truth2func_ids[ground_truth] = [len(self.func_datas)-1]
                else:
                    self.ground_truth2func_ids[ground_truth].append(len(self.func_datas)-1)
            print("Total get {} functions".format(str(len(self.func_datas))) )
            print("Total {} ground truths.".format(len(self.ground_truth2func_ids.keys())))
        
        del func_datas
        gc.collect()

        print("Saving cache...")
        if not os.path.exists(self.cache_path ):
            os.makedirs(self.cache_path)
        with open(self.cache_path + self.split + "_func_features.pickle", "wb") as f:
            pickle.dump(self.func_datas, f)
        with open(self.cache_path + self.split + "_ground_truth_map.pickle", "wb") as f:
            pickle.dump(self.ground_truth2func_ids, f)
        print("Cache saved.")
    
    def __len__(self):
        return len(self.func_datas)
    
    def generate_all_funcpairs(self):
        pass


    def __getitem__(self, index):
       
        source_function  = self.func_datas[index]
        

        # 加上相应限制
        ground_truth = source_function[-1]
        if self.arch_same:
            ground_truth += (source_function[4],)
        if self.compiler_same:
            ground_truth  += (source_function[5],)
        if self.opti_same:
            ground_truth += (source_function[6],)
        # print(ground_truth)
        if random.random() > self.neg_prob or len(self.ground_truth2func_ids[ground_truth]) < 2:
            label = -1
        else:
            label = 1
        
        if label == -1:
            target_index = self.get_neg_func(source_function)
        else:
            target_index= self.get_pos_func(ground_truth, index)
        
        target_function = self.func_datas[target_index]

        graph0 = self.get_input_graph(source_function[0], source_function[1])
        
        graph1 = self.get_input_graph(target_function[0], target_function[1])
        type_info0 = torch.Tensor(source_function[2])
        type_info1 = torch.Tensor(target_function[2])
        sc_info0 = torch.Tensor(source_function[3])
        sc_info1 = torch.Tensor(target_function[3])
        output = {
            "graph0": graph0, # edges, func_bb_insts, sc_embedding
            "graph1": graph1,
            "type_input0": type_info0, # prediction of type tasks
            "type_input1": type_info1,
            "sc_input0": sc_info0,
            "sc_input1": sc_info1,
            "label": label
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
    
    

    def get_pos_func(self, ground_truth, index):
        
        ground_truth_indices = self.ground_truth2func_ids[ground_truth]
        if len(ground_truth_indices) == 1:
            return index
        
        pos_index = index
        while pos_index == index:
            pos_index = random.choice(ground_truth_indices)
        return pos_index

    def get_neg_func(self, source_function):
        
        neg_index = random.randrange(len(self.func_datas))

        while self.func_datas[neg_index][-1] == source_function[-1]:
            neg_index = random.randrange(len(self.func_datas))
        return neg_index
    
    


    



def get_dataset(data_file, split, tokenizer, limit, opt):
    return FuncPairDataset(data_file, split, tokenizer,  limit, opt)


def get_data_loader(data_file, split,  tokenizer, limit, opt, batch_size, shuffle=True,  num_workers=0, collate_fn = None):
    dataset = FuncPairDataset(data_file, split,tokenizer, limit, opt)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=None)
    return loader

if __name__=="__main__":
    pass