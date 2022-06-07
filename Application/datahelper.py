# encoding:utf-8
import sqlite3
import pickle,os,sys
from collections import defaultdict
from random import randint
from numpy import random
from copy import deepcopy
import random as rd
from tqdm import tqdm
import numpy as np
import hashlib
from multiprocessing import Pool
import time
import logging
import json
l = logging.getLogger("datahelper.py")
logpath = os.path.join(os.path.dirname(__file__), "log")
if not os.path.exists(logpath):
    os.mkdir(logpath)
l.addHandler(logging.FileHandler(os.path.join(logpath, "datahelper.log")))
l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)
'''
function:
load dbs, query data
database schema
    sql ="""create table if not exists function (
                        func_name varchar(255),
                        bin_path varchar(255),
                        func_pick_dumps text,
                        func_embedding text,
                        is_cve integer not null defualt 0, 
                        cve_id varchar(10),
                        cve_description text,
                        cve_references text,
                        constraint pk primary key (func_name, bin_path)
                )"""
'''
class DataHelper:
    def __init__(self):
        self.db = None
        from difflib import SequenceMatcher
        self.sequecematcher = SequenceMatcher()

    def load_database(self, filename):
        '''
        :param filename: sqlite database file path
        :return: db connection
        '''

        return sqlite3.connect(filename)

    def do_query(self, db_path,  where="", limit=""):
        '''
        :param db_path:
        :param optimization:
        :return: A generator
        '''
        db = self.load_database(db_path)
        sql = "select func_name, bin_path, is_cve, cve_id, cve_description, cve_references, func_pick_dumps from function " \
             + where + " " +limit
        l.info(sql)

        cur = db.cursor()
        lines = cur.execute(sql)
        
        for line in lines:
            yield list(line) # tuple
        cur.close()
        db.close()

    def get_functions(self, db_path, start=-1, end=-1, where_suffix=None):
        '''
        :param db_path:
        :param start: query start offset
        :param end: query end offset
        :param where_suffix: where suffix
        :return: A generator : queried function info generator
        '''
        db = self.load_database(db_path)
        suffix = ""
        sql = """select func_name, bin_path, bin_name, is_cve, cve_id, cve_description, cve_references,
                func_pick_dumps from function """
        if start < 0 :
            if where_suffix:
                suffix = where_suffix
        else:
            suffix = " limit %d,%d" % (start, end-start)
        sql += suffix
        l.info("[Query]"+sql)
        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            func_pickle_dumps = line[-1]
            func = pickle.loads(func_pickle_dumps, encoding="utf-8")
            # print(func)
            yield line[:-1], func
        cur.close()
        db.close()
    
    def get_function_embeddings(self, db_path, start=-1, end=-1, where_suffix=None):
        '''
        :param db_path:
        :param start: query start offset
        :param end: query end offset
        :param where_suffix: where suffix
        :return: A generator : queried function info generator
        '''
        db = self.load_database(db_path)
        suffix = ""
        sql = """select func_name, bin_path, bin_name, is_cve, cve_id, cve_description, cve_references, func_embedding
             from function """
        if start < 0 :
            if where_suffix:
                suffix = where_suffix
        else:
            suffix = " limit %d,%d" % (start, end-start)
        sql += suffix
        l.info("[Query]"+sql)
        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            yield line[:-1], json.loads(line[-1])
        cur.close()
        db.close()

    def loads_func_info(self, func_pickle_dumps):
        return pickle.loads(func_pickle_dumps.encode("utf-8"), encoding="latin1")

    def recvore_func_info(self, data, idx):
        '''
        :param data: data[idx] pick.dumps stored strings
        :return: 将data[idx] pick.loads func_info dict
        '''
        for d in data:
            d[idx] = self.loads_func_info(d[idx])
        return data

if __name__ == '__main__':
    pass