import pandas as pd
from pathlib import Path
from torch import nn
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, BertModel
from torch.utils.data import Dataset, DataLoader
import random
import copy
import numpy as np
import torch
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
import pickle
from data import Function

print("Loading func datas and generate vocab file...")
data_file = "path_to_data_pickle"
assert os.path.exists(data_file)

if data_file.endswith(".pickle"):
    # 效率高，内存开销大
    print("Reading functions from an aggreated pickle file")
    data = []
    with open(data_file, "rb") as f:
        func_datas = pickle.load(f)
        print("Total loading {} functions".format(str(len(func_datas))) )
        for func_data in tqdm(func_datas):
            # print(func_data)
            func_bb_insts = Function(func_data).get_bb_insts()
            data.append( " ".join(func_bb_insts).strip())
    
    from collections import defaultdict
    def get_dict(data):
        words_dict = defaultdict(int)
        for i in tqdm(range( len(data) )):
            text = data[i].split()
            for c in text:
                words_dict[c] += 1
        return words_dict
    print("Generating vocab...")
    word_dict = get_dict(data)
    min_count = 0
    word_dict =  {i: j for i, j in word_dict.items() if j >= min_count}
    word_dict = dict(sorted(word_dict.items(), key=lambda s: -s[1]))
    word_dict = list(word_dict.keys())
    special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
    WORDS = special_tokens + word_dict
    print("Total {} tokens.".format(len(WORDS)))
    pd.Series(WORDS).to_csv('./vocab/Bert-vocab.txt', header=False,index=0)
    print("Vocab saved.")

elif data_file.endswith(".txt"):
    print("Reading functions from an input list...")
    word_dict = {}

    from collections import defaultdict
    def update_dict(data, word_dict):
        for i in range( len(data) ):
            text = data[i].split()
            for c in text:
                if c in word_dict:
                    word_dict[c] += 1
                else:
                    word_dict[c] = 1
    
    cur_data = []
    binary_list = []
    with open(data_file, "r") as f:
        binary_list = f.readlines()
    for binary_file in tqdm(binary_list):
        pickle_file = binary_file.strip(os.linesep)+".pickle"
        with open(pickle_file, "rb") as f:
            cur_func_datas = pickle.load(f)
            for func_data in cur_func_datas:
                func_bb_insts = Function(func_data).get_bb_insts()
                cur_data.append( " ".join(func_bb_insts).strip())
            update_dict(cur_data, word_dict)
    
    print("Generating vocab...")
    min_count = 0
    word_dict =  {i: j for i, j in word_dict.items() if j >= min_count}
    word_dict = dict(sorted(word_dict.items(), key=lambda s: -s[1]))
    word_dict = list(word_dict.keys())
    special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
    WORDS = special_tokens + word_dict
    print("Total {} tokens.".format(len(WORDS)))
    pd.Series(WORDS).to_csv('./vocab/Bert-vocab.txt', header=False,index=0)
    print("Vocab saved.")
else:
    print("The data file should be a pickle or an input list.")




