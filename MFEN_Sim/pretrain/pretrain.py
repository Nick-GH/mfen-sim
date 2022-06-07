# encoding:utf-8
import argparse
import sys
sys.path.append("..")
from dataclasses import dataclass
from gc import callbacks
import pandas as pd
from pathlib import Path
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, BertTokenizer, BertModel
from MFEN_Sim.config.pretrain_config import DataArguments, CustomTrainingArguments, ModelArguments
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.file_utils import logger, logging
from transformers.data.data_collator import default_data_collator
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_utils import EvalPrediction
from typing import List, Dict
import random
import copy
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import json
from sklearn.model_selection import train_test_split
from MFEN_Sim.dataset.data_loader_vocab import get_data_loader, get_dataset
logger.setLevel(logging.INFO)


data_args = DataArguments()
train_args = CustomTrainingArguments()
model_args = ModelArguments()


vocab_file = data_args.vocab_file
train_data_file = data_args.train_file
valid_data_file = data_args.dev_file
patience = train_args.patience


tokenizer = BertTokenizer(vocab_file)
vocab = tokenizer.get_vocab()


if train_args.load_checkpoint:
    print("Load checkpoint from {}".format(train_args.load_checkpoint_dir))
    model = BertForPreTraining.from_pretrained(train_args.load_checkpoint_dir)
else:
    config = BertConfig(vocab_size=len(vocab), 
        hidden_size= model_args.hidden_size, # H
        num_attention_heads=model_args.num_attention_heads, # A
        intermediate_size=model_args.intermediate_size,
        num_hidden_layers=model_args.num_hidden_layers, # L
        hidden_dropout_prob=model_args.hidden_dropout_prob
        )
    model = BertForPreTraining(config = config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = model.to(device)
print(model)

print("loading train dataset...")
train_data = get_dataset(train_data_file, "train", tokenizer, data_args.limit, data_args)
print("loading valid dataset...")
valid_data= get_dataset(valid_data_file, "valid", tokenizer, data_args.limit, data_args)


training_args = TrainingArguments(output_dir=train_args.checkpoint_dir,  
                                      num_train_epochs=train_args.epoch,
                                      logging_first_step = True,
                                      do_eval=train_args.do_eval,  
                                      evaluation_strategy="steps",  
                                      per_device_train_batch_size=train_args.train_batch_size,
                                      per_device_eval_batch_size=train_args.eval_batch_size,
                                      learning_rate=train_args.lr,
                                      fp16=True,
                                      save_total_limit=10,
                                      logging_steps = 1000,
                                      eval_steps=100000,
                                      save_steps = 1000,
                                      load_best_model_at_end = True,
                                      )

trainer = Trainer(model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                callbacks = [EarlyStoppingCallback(train_args.patience)])



res = trainer.evaluate(eval_dataset=valid_data)
print(res)



