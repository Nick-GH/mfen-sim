# encoding=utf-8

import os,sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
import logging
import torch, numpy
import gevent, datetime
from math import exp
from application_config import ApplicationArguments
import time

from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity
from datahelper import DataHelper
from torch_geometric.data import Data
from dataloader.data import Function
from dataloader.func_encoder import get_data_loader
from tqdm import tqdm
logger = logging.getLogger("application.py")
logpath = os.path.join(os.path.dirname(__file__), "log")
if not os.path.exists(logpath):
    os.mkdir(logpath)
logger.addHandler(logging.FileHandler(os.path.join(logpath, "application.log")))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
from multiprocessing import Pool, cpu_count
import json
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DB_CONNECTION = None # global database connection object

datahelper = DataHelper()
class Application():
    '''
    This class loads the trained model and use the model to encode an function info into a vector and calculate the similarity between functions.
    '''
    def __init__(self, config, cuda = False, model_name = "mfen" , model = None):
        '''
        :param load_path: the path to saved model. e.g. "saved_model.pt"
        :param cuda: bool : whether GPU is utilized when model calculation. Notice that you need install the torch gpu version.
        :param model_name: the model type used. Only the "bert_gnn" is supported in this script.
        :param threshold: decided what similarity scores are outputted.
        :param model: A network variant of bert_gnn
        '''
        self.config = config
        self.model = None
        self.codebert = None
        self.type_model = None
        self.model_name = model_name

        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(self.device)
        if model is not None:
            self.model = model
        else:
            if model_name=="MFEN":
                from Application.model import mfen
                self.model = mfen.MFEN(config)
        self.load_model(config.checkpoint_path, self.model)
    
    def load_model(self, path, model):
        '''
        :param path: path to saved model
        :param model: the model loaded
        :return:
        '''
        if not os.path.isfile(path):
            print("model path %s non-exists" % path)
            raise Exception
        checkpoint_state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict = checkpoint_state_dict, strict=False)
        model.eval()
        self.model =model
        self.model.to(self.device)

        from model.bert_cls import BertForClassification
        self.type_model = BertForClassification.from_pretrained(self.config.finetune_model).to(self.device)
        self.type_model.eval()
        self.codebert = BertModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.codebert.eval()
    
    
    def generate_type_embedding(self, model, x, include = True):
        total_clr_num = sum(model.config.clr_nums)
        output = torch.zeros(x["input_ids"].shape[0], total_clr_num).to(self.device)
        
        if not include:
            return output
        
        x = {k:x[k].to(self.device) for k in x}
        x = model(**x)
        type_logits = x.logits # batch_size * task_num * logits_num(11), list task_num, tensor(batch_size, logit_num)
        all_logits = type_logits[0]
        for i in range(1, len(type_logits)):
            all_logits = torch.cat([all_logits, type_logits[i]], dim=-1)
        # print(output)
        return all_logits
    
    def generate_sc_embedding(self, model, x, include = True):
        if not include:
            return torch.zeros(x["input_ids"].shape[0], model.config.hidden_size).to(self.device)
        x = {k:x[k].to(self.device) for k in x}
        output = model(input_ids = x["input_ids"], attention_mask = x["attention_mask"])
        x = output.last_hidden_state
        x = x[:,0,:]
        return x
    

    def encode_function(self, func_infos):
        '''
        :param func_info: dicts of function info
        :return: numpy vectors
        '''
        print("Encoding functions by model...")
        outputs = torch.zeros(len(func_infos), self.config.hidden_dim * 3)
        func_encoder = get_data_loader(self.config, func_infos, batch_size=1)
        with torch.no_grad():
            i = 0
            for func_data in tqdm(func_encoder):
                graph = func_data["graph"].to(self.device)
                type_input = func_data["type_input"]
                type_input = self.generate_type_embedding(self.type_model, type_input, include=self.config.use_type)
                func_sc_ids = func_data["sc_input"]
                func_sc_ids = self.generate_sc_embedding(self.codebert, func_sc_ids, include=self.config.use_string_and_const)
                output = self.model(graph, type_input, func_sc_ids, return_encoder_embedding=True)
                size = output.shape[0]
                outputs[i:i+size,:] = output
                i += size
            return outputs.detach().cpu().numpy() # embedding
    
    def encode_function_test(self, func_info):
        '''
        :param func_info: An dict of function info
        :return: a numpy vector
        '''
        # print("Encoding function: ", func_info.func_name)
        output = torch.randn(len(func_info), self.config.output_dim)
        output = output.detach().squeeze(0).cpu().numpy()
        # print(output.shape)
        return output # embedding

    def similarity_function_embedding(self, func_embedding1, func_embedding2):
        '''
        :param func_embedding1: function encoding vector
        :param func_embedding2: function encoding vector
        :return:
        '''
        sim = self.similarity_vec(func_embedding1, func_embedding2)

        return sim

    def similarity_function(self, func1, func2):
        '''
        :param func1: func1
        :param func2: func2
        :return:
        '''

        sim = self.similarity_func_info(func1, func2)
        return sim

    def similarity_func_info(self, func1, func2):
        '''
        calculate the similarity of two functions
        :param func1:  query function
        :param func2:  target function
        :return:
        '''
        func_embedding1 = self.encode_function(func1)
        func_embedding2 = self.encode_function(func2)
        res = self.similarity_vec(func_embedding1, func_embedding2)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return res

    def similarity_vec(self, func_embedding1, func_embedding2):
        '''
        calculate the similarity of  function embeddings
        :param func_embedding1: numpy.ndarray or torch.tensor
        :param func_embedding2:
        :return: similairty score ranges -1~1.
        '''
        if type(func_embedding1) is list:
            func_embedding1 = numpy.array(func_embedding1)
            func_embedding2 = numpy.array(func_embedding2)
        if type(func_embedding1) is numpy.ndarray:
            func_embedding1 = torch.from_numpy(func_embedding1).to(self.device).float()
            func_embedding2 = torch.from_numpy(func_embedding2).to(self.device).float()
        res = cosine_similarity(func_embedding1, func_embedding2)
        return res # 1

    def get_conn(self, db):
        # return the database connection
        global DB_CONNECTION
        if DB_CONNECTION is None:
            global datahelper
            DB_CONNECTION = datahelper.load_database(db)
        return DB_CONNECTION
    
    # db encode: encode_function_in_db (data_helper.get_functions) -> encode_and_update
    def encode_function_in_db(self, db_path, table_name="function"):
        '''
        Encode the functions in db file "db_path" and save the encoding vectors into table 'table_name'
        :param db_path: path to sqlite database
        '''
        db_conn = self.get_conn(db_path)
        cur = db_conn.cursor()
        sql_create_new_table = """create table if not exists %s (
                        func_name varchar(255),
                        bin_path varchar(255),
                        bin_name varchar(255),
                        func_pick_dumps text,
                        func_embedding text,
                        is_cve integer not null default 0, 
                        cve_id varchar(20),
                        cve_description text,
                        cve_references text,
                        constraint pk primary key (func_name, bin_path)
                    )""" % table_name
        try:
            cur.execute(sql_create_new_table)
            db_conn.commit()
        except Exception as e:
            logger.error("sql [%s] failed" % sql_create_new_table)
            logger.error(e)
        finally:
            cur.close()
        to_encode_list = []
        global datahelper
        # (id, func_name, bin_path, bin_name, is_cve, cve_id, cve_description, cve_references,) func_data: dict
        for func_info in datahelper.get_functions(db_path):
            to_encode_list.append(func_info[-1])
        encode_group = to_encode_list
        logger.info("Encoding for %d functions" % len(encode_group))
        self.encode_and_update(db_path, encode_group, table_name)
    
    def encode_and_update(self, db_path, functions, table_name):
        '''
        encode func infos (bb, cfg, sc, type) of the functions into vectors
        :param functions: a list contains functions to be encoded
        :param table_name: the table to save function embeddings
        '''
        db_conn = self.get_conn(db_path)
        res = []
        count = 1
        # func_embeddings = self.encode_function_test(functions)
        func_embeddings = self.encode_function(functions)
        for i,function in tqdm(enumerate(functions)):
            # len(func_info) * output_dim
            res.append( (func_embeddings[i], function["name"], function["bin_path"]) )
            count+=1
        
        result = []
        try:
            logger.info("Fetching encode results!")
            for idx, r in tqdm(enumerate(res)):
                result.append((json.dumps(r[0].tolist()), r[1], r[2]))
            logger.info("All encode fetched!")
        except Exception as e:
            print("Exception when fetching {}".format(str(e)))
        try:
            logger.info("Writing encoded vectors to database")
            cur = db_conn.cursor()
            sql_update = """ 
            update {} set func_embedding=? where func_name=? AND bin_path=?
            """.format(table_name)
            cur.executemany(sql_update, result)
            cur.close()
            db_conn.commit()
            logger.info("Writing encoded vectors to database done")
        except Exception as e:
            db_conn.rollback()
            print("Error when INSERT [{}]\n".format(sql_update))
            print(e)
        #cur.close()
    
    def read_cve_csv(self, csv_path):
        file = pd.read_csv(csv_path, encoding="gbk")
        df = pd.DataFrame(file)
        cve_datas = {}
        for i in range(len(df)):
            func_names = df.iloc[i]["FunctionName"].strip().split(",")
            versions = df.iloc[i]["Version"].strip().split(",")
            for func_name in func_names:
                for version in versions:
                    # print(func_name, df.iloc[i]["Description"], df.iloc[i]["References"], df.iloc[i]["Name"], df.iloc[i]["Package"], version)
                    cve_datas[(func_name, df.iloc[i]["Package"]+"-"+version)] = (df.iloc[i]["Name"], df.iloc[i]["Description"], df.iloc[i]["References"])
        return cve_datas
    
    def delete_non_cve(self, db_path):
        db_conn = self.get_conn(db_path)
        sql_delete = "delete from function where is_cve = 0"
        cursor = db_conn.cursor()
        cursor.execute(sql_delete)
        db_conn.commit()
        db_conn.close()

    def update_cve_info_by_csv(self, db_path, csv_path, delete = False):
        
        update_datas = []
        cve_datas = self.read_cve_csv(csv_path)
        # print(len(cve_datas))
        global datahelper
        # print(db_path)
        functions = datahelper.do_query(db_path)
        try:
            logger.info("Matching cve info!")
            for function in functions:
                func_name = function[0]
                bin_path = function[1]
                splits = bin_path.split(os.sep)
                for package_version in splits:
                    if (func_name, package_version) in cve_datas:
                        cve_id, cve_description, cve_references = cve_datas[(func_name, package_version)]
                        update_datas.append((1, cve_id, cve_description, cve_references, func_name, bin_path))
            logger.info("Total match {} cve functions.".format(len(update_datas)))
        except Exception as e:
            print("Exception when matching cve info {}".format(str(e)))
        
        db_conn = self.get_conn(db_path)
        try:
            logger.info("Writing cve info to database")
            sql_update = """ 
                    update function set is_cve=?, cve_id=?, cve_description=?, cve_references=? where func_name = ? and bin_path = ?
                    """
            cursor = db_conn.cursor()
            cursor.executemany(sql_update, update_datas)
            
        except Exception as e:
            db_conn.rollback()
            print("Error when INSERT [{}]\n".format(sql_update))
            print(e)
        cursor.close()
        db_conn.commit()
        db_conn.close()
        if delete:
            self.delete_non_cve(db_path)

def parse_args_in_app():
    '''
    :return:The args in dict
    '''
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--db_path", type=str,
                    help="the path to sqlite db file where functions are saved")
    ap.add_argument("--csv_path", type=str,
                    help="the path to cve csv file where cve functions are saved")
    ap.add_argument("--delete_non_cve", action="store_true",
                    help="whether delete functions which are not cve functions")
    return ap.parse_args()


# insert CVEs according to the csv file
# 1. extract functions from binaries and stored into db
# 2. read the csv file and update the csv fields
# 3. delete all function that is_cve == 0


if __name__ == '__main__':
    args = parse_args_in_app()
    application_args = ApplicationArguments()
    print(args)
    app = Application(application_args, application_args.cuda)
    if args.db_path:
        app.encode_function_in_db(args.db_path)
    if args.csv_path:
        app.update_cve_info_by_csv(args.db_path, args.csv_path, args.delete_non_cve)
        
