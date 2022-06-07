# coding:utf-8
import enum
import os
import sys
sys.path.append("..")
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
from transformers import BertTokenizer,  AutoTokenizer, BertModel
from transformers.models.bert.tokenization_bert import load_vocab

from MFEN_Sim.dataset import data_loader_cls, data_loader_feature, data_loader_func
from MFEN_Sim.model.bert_cls import BertForClassification

import pandas as pd
from sklearn.model_selection import train_test_split
from MFEN_Sim.config import type_config, embedding_generator_config
from torch_scatter import scatter



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generate signature prediction embeddings and code literal embeddings for functions before similarity training

def generate_all_sp_embedding(train_args, data_args, output_path):
    vocab_file = data_args.vocab_file
    print(train_args.best_dir)
    type_model = BertForClassification.from_pretrained(train_args.best_dir).to(device)
    vocab = load_vocab(vocab_file)
    # print(vocab)
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=True, never_split=vocab)
    
    train_data_loader = data_loader_cls.get_data_loader(data_args.train_file, "train", tokenizer, -1, data_args, 768, shuffle=False, num_workers=8)
    generate_sp_embedding(type_model, train_data_loader, output_path, "train")
    valid_data_loader = data_loader_cls.get_data_loader(data_args.valid_file, "valid", tokenizer, -1, data_args, 768, shuffle=False, num_workers=8)
    generate_sp_embedding(type_model, valid_data_loader, output_path, "valid")
    test_data_loader = data_loader_cls.get_data_loader(data_args.test_file, "test", tokenizer, -1, data_args, 768, shuffle=False, num_workers=8)
    generate_sp_embedding(type_model, test_data_loader, output_path, "test")

def generate_sp_embedding(model, data_loader, output_path, split="train"):
    print("{} signature prediction embedding generating...".format(split))
    
    print(len(data_loader.dataset))
    total_clr_num = sum(model.config.clr_nums)
    output_tensor = torch.zeros(len(data_loader.dataset), total_clr_num).to(device)
    model.eval()
    with torch.no_grad():
        i = 0
        for data in tqdm(data_loader):
            data = {k:data[k].to(device) for k in data}
            # print(data["labels"])
            output = model(**data)
            type_logits = output.logits # batch_size * task_num * logits_num(11), list
            size = type_logits[0].shape[0]
            # print(size)
            all_logits = type_logits[0]
            for j in range(1,len(type_logits)):
                all_logits = torch.cat([all_logits, type_logits[j]], dim=-1)
            
            output_tensor[i:i+size,:] = all_logits
            i += size
    print(output_tensor[0])
    print("{} signature prediction embedding generated.".format(split))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(output_tensor.to("cpu").float(), output_path+split+"_sp_embeddings.pt")

def generate_all_cl_embedding(data_args, output_path):
    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = BertModel.from_pretrained("microsoft/codebert-base").to(device)
    batch_size = 64

    train_data_loader = data_loader_feature.get_data_loader(data_args.train_file, "train", codebert_tokenizer, -1, data_args, batch_size, shuffle=False, num_workers=4)
    generate_cl_embedding(codebert, train_data_loader, output_path, "train")
    valid_data_loader = data_loader_feature.get_data_loader(data_args.valid_file, "valid", codebert_tokenizer, -1, data_args, batch_size, shuffle=False, num_workers=4)
    generate_cl_embedding(codebert, valid_data_loader, output_path, "valid")
    test_data_loader = data_loader_feature.get_data_loader(data_args.test_file, "test", codebert_tokenizer, -1, data_args, batch_size, shuffle=False, num_workers=4)
    generate_cl_embedding(codebert, test_data_loader, output_path, "test")



def generate_cl_embedding(model, data_loader, output_path, split="train"):
    print("{} code literal embedding generating...".format(split))
    print(len(data_loader.dataset))
    model.eval()
    all_func_feature_embedding = torch.zeros(len(data_loader.dataset), model.config.hidden_size).to(device)
    with torch.no_grad():
        i = 0
        for data in tqdm(data_loader):
            # data = {k:data[k].to(device) for k in data}
            data["input_ids"] = data["input_ids"].to(device)
            data["attention_mask"] = data["attention_mask"].to(device)
            output = model(input_ids = data["input_ids"], attention_mask = data["attention_mask"])
            x = output.last_hidden_state
            x = x[:,0,:] # batch_size * dim
            size = x.shape[0]
            # print(x.shape)
            all_func_feature_embedding[i:i+size] = x
            # print(all_func_feature_embedding[i:i+size])
            i += size
            # print(len(all_func_bb_feature_embedding))
    print(all_func_feature_embedding.shape) # func num
    print(all_func_feature_embedding)
    print("{}  code literal embedding generated.".format(split))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(all_func_feature_embedding.to("cpu"), output_path+split+"_cl_embeddings.pt")





if __name__ == "__main__":

    generate_all_sp_embedding(type_config.CustomTrainingArguments(), embedding_generator_config.TypeDataArguments(), 
                                "cache/embeddings/")

    generate_all_cl_embedding(embedding_generator_config.CodeLiteralDataArguments(), 
                                 "cache/embeddings/")

