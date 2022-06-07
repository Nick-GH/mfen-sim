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
import joblib
from data import Function



data_file = "pickle, input_list.txt, "
output_path = ""
assert os.path.exists(data_file)
split=10
print("Split training dataset...")
if os.path.isdir(data_file):
    pickle_list = []
    for file in os.listdir(data_file):
        if file.endswith(".pickle"):
            pickle_list.append(os.path.join(data_file, file))
    # print(pickle_list)

    func_datas = []
    for pickle_file in pickle_list:
        with open(pickle_file, "rb") as f:
            cur_func_datas = pickle.load(f)
            func_datas.extend(cur_func_datas)
    print("Total loading {} functions".format(str(len(func_datas))) )

    test_size = 0.2
    
    train_datas, eval_datas = train_test_split(func_datas, test_size=test_size, shuffle=True)

    valid_datas, test_datas = train_test_split(eval_datas, test_size=0.5, shuffle=True)
    print("Train {}, valid {}, test {}".format(len(train_datas), len(valid_datas), len(test_datas)))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    data_size = len(train_datas)//split
    chunk = 0
    for i in range(0, len(train_datas), data_size):
        chunk += 1
        data_path = os.path.join(output_path, str(chunk)+"_train_data.pickle")
        tmp_train_datas = train_datas[i:min(i+data_size, len(train_datas))]
        with open(data_path, "wb") as f:
            print("Saving func datas...")
            pickle.dump(tmp_train_datas, f)
            print( "Saving {} done.".format(str(chunk)) )
        pass

    with open("{}/valid_data.pickle".format(output_path), "wb") as f:
        pickle.dump(valid_datas, f)

    with open("{}/test_data.pickle".format(output_path), "wb") as f:
        pickle.dump(test_datas, f)


elif data_file.endswith(".pickle"):
    print("Reading functions from an aggreated pickle file")
    assert os.path.exists(data_file)
    func_datas = []
    with open(data_file, "rb") as f:
        func_datas = pickle.load(f)
        print("Total loading {} functions".format(str(len(func_datas))) )
    
    test_size = 0.2
    
    train_datas, eval_datas = train_test_split(func_datas, test_size=test_size, shuffle=True)

    valid_datas, test_datas = train_test_split(eval_datas, test_size=0.5, shuffle=True)
    print("Train {}, valid {}, test {}".format(len(train_datas), len(valid_datas), len(test_datas)))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("{}/train_data.pickle".format(output_path), "wb") as f:
        pickle.dump(train_datas, f)

    with open("{}/valid_data.pickle".format(output_path), "wb") as f:
        pickle.dump(valid_datas, f)

    with open("{}/test_data.pickle".format(output_path), "wb") as f:
        pickle.dump(test_datas, f)
    
elif data_file.endswith(".txt"):

    
    print("Reading functions from an input list...")
    if not os.path.exists("./data_file"):
        os.makedirs("./data_file")

    with open("{}/train_data.pickle".format(output_path), "wb") as f:
        print("Clear")
    with open("{}/valid_data.pickle".format(output_path), "wb") as f:
        print("Clear")
    with open("{}/test_data.pickle".format(output_path), "wb") as f:
        print("Clear")

    cur_data = []
    binary_list = []
    test_size = 0.1
    all_train_datas = []
    all_valid_datas = []
    all_test_datas = []
    with open(data_file, "r") as f:
        binary_list = f.readlines()
        for binary_file in tqdm(binary_list):
            file = binary_file.strip()+".pickle"
            with open(file, "rb") as f:
                cur_func_datas = pickle.load(f)
                print("Total loading {} functions in cur func data".format(str(len(cur_func_datas))) )
                if len(cur_func_datas) < 10:
                    continue
                train_datas, eval_datas = train_test_split(cur_func_datas, test_size=test_size, shuffle=True)

                valid_datas, test_datas = train_test_split(eval_datas, test_size=0.5, shuffle=True)
                all_train_datas.extend(train_datas)
                all_valid_datas.extend(valid_datas)
                all_test_datas.extend(test_datas)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("{}/train_data.pickle".format(output_path), "wb") as f:
        pickle.dump(train_datas, f)

    with open("{}/valid_data.pickle".format(output_path), "wb") as f:
        pickle.dump(valid_datas, f)

    with open("{}/test_data.pickle".format(output_path), "wb") as f:
        pickle.dump(test_datas, f)

            
else:
    print("The data file should be a pickle or an input list.")

