# coding:utf-8
import enum
import sys
sys.path.append("..")
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
import pandas as pd
import copy
import gc
from sklearn.model_selection import train_test_split
from collections import defaultdict
import networkx as nx
import time
import sys
from MFEN_Sim.config.filter_config import DataArguments



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphFilter:
    def __init__(self, file_name, split, opt) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.cache_path = opt.cache_path

        self.min_size = opt.min_size
        self.max_size = opt.max_size 

    
    def truncate_graph(self):
        print("Filtering functions ...")
        with open(self.data_dir, "rb") as f:
            func_datas = pickle.load(f)
        print("Total {} functions.".format(len(func_datas)))
        filtered_func_datas = []
        count = 0
        for func_data in tqdm(func_datas):
            g = nx.DiGraph()
            cfg_size = func_data["cfg_size"]
            if cfg_size >= self.min_size and cfg_size <= self.max_size:
                filtered_func_datas.append(func_data)
                continue
            if cfg_size < self.min_size:
                continue
            cfg_edges = func_data["cfg"]
            prev_edge_count = len(cfg_edges)
            g.add_edges_from(cfg_edges)
            func_data["bb_data"] = func_data["bb_data"][: self.max_size]
            remove_nodes = [i for i in range(self.max_size, cfg_size)]
            g.remove_nodes_from(remove_nodes)
            func_data["cfg"] = list(g.edges)
            func_data["prev_cfg_size"] = cfg_size
            func_data["cfg_size"] = len(func_data["bb_data"])
            print("Prev nodes {}, edges {}, cur nodes {}, edges {}.".format(cfg_size, 
                prev_edge_count, 
                len(func_data["bb_data"]), 
                len(func_data["cfg"]), ))
            count += 1
            filtered_func_datas.append(func_data)
        print("Total {} functions left.".format(len(filtered_func_datas)))
        print("Total turncate {} functions, {}".format(count, count/len(func_datas)))
        with open(self.cache_path + self.split + "_filtered_functions.pickle", "wb") as f:
            print("Saving cache...")
            pickle.dump(filtered_func_datas, f)
            print("Cache saved.")
        return filtered_func_datas

def filter_graphs(file_name, split, data_args):
    graph_filter = GraphFilter(file_name, split, data_args)
    graph_filter.truncate_graph()

if __name__ == "__main__":
    data_args = DataArguments()
    filter_graphs(data_args.train_file,  "train", data_args,)
    filter_graphs(data_args.valid_file,  "valid", data_args, )
    filter_graphs(data_args.test_file,  "test", data_args,)