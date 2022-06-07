import argparse
import sys
sys.path.append("..")
import pandas as pd
from pathlib import Path
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, BertModel, BertTokenizer, EarlyStoppingCallback
from transformers.models.bert.tokenization_bert import load_vocab
from transformers.trainer_utils import EvalPrediction
from MFEN_Sim.model.bert_cls import BertForClassification
from torch.utils.data import Dataset, DataLoader
from MFEN_Sim.config.type_config import DataArguments, CustomTrainingArguments, ModelArguments
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
from MFEN_Sim.dataset.data_loader_cls import get_data_loader, get_dataset
from typing import List, Dict
from transformers.file_utils import logger, logging
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logger.setLevel(logging.INFO)





device = 'cuda' if torch.cuda.is_available() else 'cpu'
def run(model_args, data_args, args):


    training_args = TrainingArguments(output_dir=args.checkpoint_dir, 
                                      overwrite_output_dir=True, 
                                      weight_decay=0.1,
                                      num_train_epochs=args.epoch,
                                      # disable_tqdm=True,
                                      learning_rate=args.lr,
                                      logging_dir="./runs",
                                      do_eval=args.do_eval, 
                                      evaluation_strategy="steps",  
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      load_best_model_at_end=True,  
                                      metric_for_best_model="overall_acc", 
                                      save_total_limit=10,
                                      fp16=True,
                                      logging_first_step=True,
                                      logging_steps = 1000,
                                      eval_steps = 5000,
                                      save_steps = 5000,
                                      )
    vocab_file = data_args.vocab_file
    train_data_file = data_args.train_file
    valid_data_file = data_args.valid_file
    test_data_file = data_args.test_file

    vocab = load_vocab(vocab_file=vocab_file)
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=True, never_split=vocab)

    train_data = get_dataset(train_data_file, "train", tokenizer, data_args.limit, data_args)
    valid_data = get_dataset(valid_data_file, "valid", tokenizer, data_args.limit, data_args)
    test_data= get_dataset(test_data_file, "test", tokenizer, data_args.limit, data_args)

    if args.load_checkpoint:
        print("Load checkpoint from {}".format(args.load_checkpoint_dir))
        model = BertForClassification.from_pretrained(args.load_checkpoint_dir, model_args=model_args)
    else:
        print("Load checkpoint from {}".format(model_args.pretrained_model))
        model = BertForClassification.from_pretrained(model_args.pretrained_model, model_args=model_args)
    
    print(model)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=valid_data,
                      compute_metrics=multi_cls_metric,
                      callbacks = [EarlyStoppingCallback(args.patience)])
    
    trainer.train()
    print("test dataset eval")
    logger.info(trainer.evaluate(test_data))
    trainer.save_model(args.best_dir)
    print("train dataset eval")
    logger.info(trainer.evaluate(train_data))


def multi_cls_metric(eval_output: EvalPrediction) -> Dict[str, float]: 
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """
    """
    该函数是回调函数,Trainer会在进行评估时调用该函数
    """
    all_preds = eval_output.predictions # task_num, batch_size, label_num
    all_labels = eval_output.label_ids # tensor(batch_size, task_num)
    
    # print(all_labels.shape)
    overall_acc = 0.0
    metrics = dict()
    for i in range(len(all_preds)):
        # print(len(all_preds))
        preds = all_preds[i] # batch_size, label_num
        labels = all_labels[:,i]

        preds = np.argmax(preds, axis=-1).flatten()
        labels = labels.flatten()
        acc = accuracy_score(labels, preds)
        metrics["task {} acc".format(i)] = acc
        overall_acc += acc
    metrics["overall_acc"] = overall_acc
    return metrics




if __name__ == "__main__":
    data_args = DataArguments()
    train_args = CustomTrainingArguments()
    model_args = ModelArguments()
    
    model_args.clr_targets = ["arg_num", "arg_1", "arg_2", "arg_3"]
    model_args.clr_nums = [11 for _ in range(len(model_args.clr_targets))]

    data_args.pred_arg_num = len(model_args.clr_targets)-1
    assert len(model_args.clr_targets) == len(model_args.clr_nums)
    run(model_args, data_args, train_args)


    

    
