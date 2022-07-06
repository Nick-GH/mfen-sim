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
from torch.utils.data import Dataset, DataLoader, dataset
from transformers.file_utils import logger, logging
from transformers import BertTokenizer, PreTrainedTokenizerFast
from tokenizers import  BertWordPieceTokenizer
from .data import Function, BasicBlock
import pandas as pd
import copy
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
import gc

# dataloader for bert pretraining
class BertDataset(Dataset):
    def __init__(self, file_name, split, tokenizer, limit=-1, opt=None) -> None:
        super().__init__()
        self.file_name = file_name
        self.split = split
        self.opt = opt
        self.data_dir = os.path.join(opt.root, self.file_name)
        self.max_length = opt.max_length
        self.max_pos_length = opt.max_pos_length

        self.bb_pairs = [] 
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.mask_prob = opt.mask_prob
        self.random_token_prob = opt.random_token_prob
        self.leave_unmasked_prob = opt.leave_unmasked_prob
        self.read_cache = opt.read_cache
        self.cache_path = opt.cache_path
        self.limit = limit

        if self.read_cache and os.path.exists(self.cache_path + self.split + "_cache.pickle"):
            print("Reading cache...")
            with open(self.cache_path + self.split + "_cache.pickle", "rb") as f:
                self.bb_pairs = pickle.load(f)
            print("Cache read.")
        else:
            self._preprocess()
        
        self.bb_pairs = self.bb_pairs[:self.limit]
        gc.collect()


    def _preprocess(self):
        print("Loading func datas...")
        assert os.path.exists(self.data_dir)
        with open(self.data_dir, "rb") as f:
            func_datas = pickle.load(f)
            print("Total loading {} functions".format(str(len(func_datas))) )
            for func_data in tqdm(func_datas):
                function = Function(func_data)
                bb_pairs = function.get_adj_bb_pair()
                if bb_pairs == None:
                    continue

                self.bb_pairs.extend(bb_pairs)
        # gc
        del func_datas
        gc.collect()
        print("Saving cache...")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(self.cache_path + self.split + "_cache.pickle", "wb") as f:
            pickle.dump(self.bb_pairs, f)
            print("Cache saved.")
    
    def __len__(self):
        return len(self.bb_pairs)
    
    def __getitem__(self, index):
        bb1, bb2, nsp_label = self.get_sentence(index)
        bb1_random, bb1_label = self.random_word(bb1.split())
        bb2_random, bb2_label = self.random_word(bb2.split())
        
        bb1 = [self.vocab['[CLS]']] + bb1_random + [self.vocab['[SEP]']]
        bb2 = bb2_random + [self.vocab['[SEP]']]
        bb1_label = [self.vocab['[PAD]']] + bb1_label + [self.vocab['[PAD]']]
        bb2_label = bb2_label + [self.vocab['[PAD]']]

        
        segment_label = ([0 for _ in range(len(bb1))] + [1 for _ in range(len(bb2))])[:self.max_pos_length]
        
        bert_input = (bb1 + bb2)[:self.max_length]
        bert_label = (bb1_label + bb2_label)[:self.max_length]

        
        padding = [self.vocab['[PAD]'] for _ in range(self.max_length - len(bert_input))]
        
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        attention_mask = np.array(attention_mask) # 
        bert_input = np.array(bert_input) # 
        segment_label = np.array(segment_label) # 
        bert_label = np.array(bert_label) # 
        is_next_label = nsp_label
        output = {"input_ids": bert_input,
                  "token_type_ids": segment_label,
                  'attention_mask': attention_mask,
                  "labels": bert_label,
                  "next_sentence_label": is_next_label}
        return output
    
    def random_word(self, sentence,):
        tokens = [char for char in sentence]

        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < self.mask_prob:
                mask_prob = random.random()
                if mask_prob < 1 - self.random_token_prob - self.leave_unmasked_prob:
                    tokens[i] = self.vocab['[MASK]']
                elif mask_prob < 1 - self.random_token_prob:
                    tokens[i] = random.randrange(len(self.vocab))
                else:
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                
                output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                
                output_label.append(-100)
        return tokens, output_label


    def get_sentence(self, index):
        '''
        next_sentence_label (torch.LongTensor of shape (batch_size,), optional) –
        Labels for computing the next sequence prediction (classification) loss. 
        Input should be a sequence pair (see input_ids docstring) Indices should be in [0, 1]:
        0 indicates sequence B is a continuation of sequence A,
        1 indicates sequence B is a random sequence.
        '''
        bb1 = self.bb_pairs[index][0]
        bb2 = self.bb_pairs[index][1]
        if random.random() > 0.5:
            
            return bb1, bb2, 0
        else:
            if random.random() > 0.5:
                return bb2, bb1, 1
            
            else:
                neg_index = index
                while neg_index == index:
                    neg_index = random.randrange(len(self.bb_pairs))
                return bb1, self.bb_pairs[neg_index][1], 1

# map style dataset
def collate_fn(features) -> Dict[str, Tensor]:
    pass

def get_dataset(data_file, split, tokenizer, limit, opt):
    return BertDataset(data_file, split, tokenizer, limit, opt)

def get_data_loader(data_file, split, tokenizer, limit, opt, batch_size,  shuffle,  num_workers=8, collate_fn = None):
    dataset = BertDataset(data_file, split, tokenizer, limit, opt)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader

if __name__=="__main__":
   pass