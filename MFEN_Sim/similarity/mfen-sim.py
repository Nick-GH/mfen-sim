import argparse
import sys
sys.path.append("..")
import pandas as pd
from pathlib import Path
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, BertModel, BertTokenizer, training_args, AutoTokenizer
from transformers.trainer_utils import EvalPrediction
from MFEN_Sim.model.mfen import MFEN
from torch.utils.data import Dataset, DataLoader
from MFEN_Sim.config.similarity_config import DataArguments, CustomTrainingArguments, ModelArguments
from MFEN_Sim.model.bert_cls import BertForClassification
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
from MFEN_Sim.dataset.data_loader_func import get_batch_sample_data_loader
from typing import List, Dict
from transformers.file_utils import logger, logging
from transformers import TrainingArguments, Trainer
from datasets import load_metric

import torch.nn.functional as F
from transformers.models.bert.tokenization_bert import load_vocab
import loss_fns


import gc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import online_metrics
logger.setLevel(logging.INFO)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def compute_metircs(distances, labels, threshold=0.5):
    preds = distances > threshold
    metrics = {}
    metrics["acc"] = accuracy_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds) 
    metrics["recall"] = recall_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)
    metrics["AUC"] = roc_auc_score(labels, distances)
    metrics["threshold"] = threshold
    return metrics


def aggregate_metrics(total_metrics):
    if len(total_metrics) == 0:
        return None
    keys = total_metrics[0].keys()
    length = len(total_metrics)
    avg_metrics = {}
    for key in keys:
        sum = 0.0
        for metric in total_metrics:
            sum += metric[key]
        avg_metrics[key] = sum / length
    return avg_metrics

def train(model, data_loader, train_args, optim, loss_func, eval_metric, cuda = True, log_interval=1000, accumulation_steps = 1, save_step = 20000):

    model.train()
    num_batches = len(data_loader)
    total_loss = 0.0
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    step = 0
    embeddings = None
    total_distances = []
    total_labels = []
    for data in tqdm(data_loader):
        if step == 0:
            optim.zero_grad()
        graph = data["graph"].to(device)
        type_input = data["type_input"].to(device)
        func_sc_ids = data["sc_input"].to(device)

        outputs = model(graph, type_input, func_sc_ids)
        if embeddings == None:
            embeddings = outputs
        else:
            embeddings = torch.cat([embeddings, outputs], dim=0)
        
        step += 1
        if step % log_interval == 0:
            print("cur train loss: ", total_loss / step)
            threshold = Find_Optimal_Cutoff(np.array(total_labels), np.array(total_distances))[0]
            cur_metrics = compute_metircs(np.array(total_distances),np.array(total_labels),  threshold)
            print("cur train metrics: ", cur_metrics)

        if step % accumulation_steps == 0:
            loss = loss_func(embeddings)
            total_loss += loss.item()

            loss.backward()
            optim.step()
            optim.zero_grad()

            distances, labels = eval_metric.metrics(embeddings, return_metric=False)
            total_distances.extend(distances)
            total_labels.extend(labels)
            embeddings = None
        
        if step % save_step == 0:
            torch.save(model.state_dict(), train_args.checkpoint_dir + "similarity_{}.pth".format(step), _use_new_zipfile_serialization=False)

    threshold = Find_Optimal_Cutoff(np.array(total_labels), np.array(total_distances))[0]
    final_metrics = compute_metircs(np.array(total_distances),np.array(total_labels), threshold)
    return total_loss/num_batches, final_metrics


def evaluation(model, data_loader, loss_func, eval_metric, cuda = True, accumulation_steps = 1):

    model.eval()
    num_batches = len(data_loader)
    total_loss = 0.0
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    step = 0
    embeddings = None
    total_distances = []
    total_labels = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            graph = data["graph"].to(device)
            type_input = data["type_input"].to(device)
            func_sc_ids = data["sc_input"].to(device)
            outputs = model(graph, type_input, func_sc_ids)

            if embeddings == None:
                embeddings = outputs
            else:
                embeddings = torch.cat([embeddings, outputs], dim=0)

            step += 1
            if step % accumulation_steps == 0:
                loss = loss_func(outputs)
                total_loss += loss.item()

                distances, labels = eval_metric.metrics(embeddings, return_metric=False)
                total_distances.extend(distances)
                total_labels.extend(labels)
                embeddings = None
        
        threshold = Find_Optimal_Cutoff(np.array(total_labels), np.array(total_distances))[0]
        final_metrics = compute_metircs(np.array(total_distances),np.array(total_labels),  threshold)
    return total_loss/num_batches, final_metrics

if __name__ == "__main__":
    data_args = DataArguments()
    train_args = CustomTrainingArguments()
    model_args = ModelArguments()
    # print(data_args)
    # print(train_args)
    # print(model_args)

    model_args.clr_targets = ["arg_num", "arg_1", "arg_2", "arg_3", "arg_4"]
    model_args.clr_nums = [11 for _ in range(len(model_args.clr_targets))]
    
    vocab_file = data_args.vocab_file
    train_data_file = data_args.train_file
    valid_data_file = data_args.valid_file
    test_data_file = data_args.test_file

    vocab = load_vocab(vocab_file=vocab_file)
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=True, never_split=vocab)
    
    # dataloader
    train_data_loader = get_batch_sample_data_loader(train_data_file, "train", tokenizer,  -1, data_args, train_args.train_batch_size, 2,  num_workers=0, collate_fn = None)
    valid_data_loader = get_batch_sample_data_loader(valid_data_file, "valid", tokenizer, -1, data_args, train_args.eval_batch_size, 2,  num_workers=0, collate_fn = None)
    test_data_loader = get_batch_sample_data_loader(test_data_file, "test", tokenizer, -1, data_args, train_args.eval_batch_size, 2, num_workers=0, collate_fn = None)

    # 加载模型
    model = MFEN(model_args)
    if train_args.load_checkpoint:
        print("Load checkpoint from {}".format(train_args.load_checkpoint_path))
        state_dict = torch.load(train_args.load_checkpoint_path)
        model.load_state_dict(state_dict=state_dict)
    print("Model created.")
    print(model)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=train_args.lr)
    criterion  = nn.CrossEntropyLoss()
    model = model.to(device)

    loss_fn = loss_fns.InfoNCELoss()
    
    metric = online_metrics.InBatchPairMetric(num_sent=2)

    if not os.path.exists(train_args.checkpoint_dir):
        os.makedirs(train_args.checkpoint_dir)
    
    if not os.path.exists(train_args.best_dir):
        os.makedirs(train_args.best_dir)

    patience = train_args.patience
    valid_loss = 9999
    print('=*'*50)

    best_loss = valid_loss
    
    for epoch in range(train_args.epoch):
        print("Epoch: {}".format(epoch+1))
        train_loss, train_metrics = train(model, train_data_loader, train_args, optim, loss_fn, metric)
        print("Epoch {} train loss: {}".format(epoch+1, train_loss))
        print("Epoch {} train metrics: {}".format(epoch+1, train_metrics))

        valid_loss, valid_metrics = evaluation(model, valid_data_loader, loss_fn, metric)
        print('=*'*50)
        print("Epoch {} valid loss: {}".format(epoch+1, valid_loss))
        print("Epoch {} valid metrics: {}".format(epoch+1, valid_metrics))
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), train_args.best_dir + f'similarityBest.pth', _use_new_zipfile_serialization=False)
        elif patience != -1:
            patience -= 1
            if patience == 0:
                print("loss doesn't reduce for {} epoches, early stop.".format(str(train_args.patience)))
                break
        print('=*'*50)
        torch.save(model.state_dict(), train_args.checkpoint_dir + 'similarity_epoch_{}.pth'.format(epoch+1), _use_new_zipfile_serialization=False)
    
    # test
    test_loss, test_metrics = evaluation(model, test_data_loader, loss_fn, metric)
    print('=*'*50)
    print('Final test loss:', test_loss)
    print("Final test metrics: ", test_metrics)







    
