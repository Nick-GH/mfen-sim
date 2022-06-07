# encoding:utf-8

# perform similarity calculation based on 'application.py'
import os, sys
from math import exp
from typing import List

from numpy.lib.utils import _set_function_name
from application_config import ApplicationArguments
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
from datahelper import DataHelper
from application import Application

from tqdm import tqdm
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import json
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import logging
from collections import defaultdict
import torch
l = logging.getLogger("main_app")
log_dir = "./log"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
l.addHandler(logging.FileHandler("./log/main_app.log"))
# l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)


class BinarySearch():
    def __init__(self, config, cuda=False):
        #cuda = True
        self.dh = DataHelper() 
        self.config = config
        self.cuda = cuda
        l.info("[I] Model Loading....")
        self.compute_app = Application(config, cuda=cuda)
        l.info("[I] Model loaded...")

    def function_embedding_similarity(self, sources = [], targets = [], threshold = 0, topk = 10):
        # (id, func_name, bin_path, bin_name, is_cve, cve_id, cve_description, cve_references, func_embedding,), Function
        '''
        :param sources:list: source function info(Function) and embeddings 
        :param targets:list: target function info(Function) and embeddings
        :return: dict: key is function_name, value is a dict :{'rank':[], 'info':(function_name, bin_path, bin_name, func_embedding)}
        '''
        result = defaultdict(dict)
        s_func_embeddings = torch.zeros(len(sources), len(sources[0][-1]))
        t_func_embeddings = torch.zeros(len(targets), len(targets[0][-1]))
        for i, (_, s_func_embedding) in enumerate(sources):
            s_func_embeddings[i,:] = torch.Tensor(s_func_embedding).float()
        for i, (_, t_func_embedding) in enumerate(targets):
            t_func_embeddings[i,:] = torch.Tensor(t_func_embedding).float()
        sims = self.compute_app.similarity_function_embedding(s_func_embeddings, t_func_embeddings) # M * N
        for i, (s_func_info, _ ) in tqdm(enumerate(sources)):
            res = []
            s_func_name, s_bin_path, s_bin_name,  s_is_cve, s_cve_id, s_cve_description, s_cve_references = s_func_info
            for j, (t_func_info, _ ) in enumerate(targets):
                # print(i, j)
                # t_func_name, t_bin_path, t_bin_name,  t_is_cve, t_cve_id, t_cve_description, t_cve_references = t_func_info
                res.append( ( t_func_info, sims[i][j]) )
            
            similarity_list = list(filter(lambda x:x[-1] >= threshold, res))
            if len(similarity_list) == 0:
                continue
            similarity_list.sort(key=lambda x: x[-1], reverse=True) # 排序
            similarity_list = similarity_list[:topk]
            rank_similarity_list = []
            for t_func_info, sim in similarity_list:
                t_func_name, t_bin_path, t_bin_name,  t_is_cve, t_cve_id, t_cve_description, t_cve_references = t_func_info
                if t_is_cve == 1:
                    rank_similarity_list.append(
                        {"FuncName": t_func_name, "BinaryPath":t_bin_path, "BinaryName": t_bin_name,
                        "CVE_ID": t_cve_id, "CVE_DESCRIPTION": t_cve_description, "CVE_REFERENCES":t_cve_references,
                        "SIM": str(sim)})
                else:
                    rank_similarity_list.append(
                        {"FuncName": t_func_name, "BinaryPath":t_bin_path, "BinaryName": t_bin_name,
                        "SIM": str(sim)})
            if s_is_cve == 1:
                s_func = {"FuncName": s_func_name, "BinaryPath":s_bin_path, "BinaryName": s_bin_name,
                        "CVE_ID": s_cve_id, "CVE_DESCRIPTION": s_cve_description, "CVE_REFERENCES":s_cve_references}
            else:
                s_func = {"FuncName": s_func_name, "BinaryPath":s_bin_path, "BinaryName": s_bin_name}
            result[s_func_name+","+s_bin_path]['info'] = s_func
            result[s_func_name+","+s_bin_path]['rank'] = rank_similarity_list
        return result
    
    def prefilter(self, func1, func2):
        '''
        :param cfg1:
        :param cfg2:
        :return: if cfg1 and cfg2 are too different , return 1.
        '''
        c1 = func1.cfg_size
        c2 = func2.cfg_size
        if abs(c1-c2) > 30:
            return 1

        if c1/c2 > 3 or c2/c1 > 3:
            return 1
        return 0
    
    def db_similarity(self, source_db, target_db, threshold, topk, start=-1, end=-1):
        '''
        :param source_db: query function database
        :param target_db: target function database
        :param use_function: True: use raw feature, False: use function embedding;
        :param threshold: float: -1~1
        :param start/end: the position for select in sql limit
        :return:
        '''
        source_funcs = []
        target_funcs = []
        bin_names = set()
        print("Use function embedding...")
        where_suffix = "" 
        # get_functions: (id, func_name, bin_path, bin_name, is_cve, cve_id, cve_description, cve_references ), func_embedding
        for func in self.dh.get_function_embeddings(source_db, where_suffix=where_suffix):
            source_funcs.append(func)
            # bin_names.add("'"+func[0][2].split('.')[0]+"%'")
        # bin_files = " or ".join(bin_names)
        # where_suffix = " where bin_file_name like %s" % bin_files
        #l.info("[DB] the firmware select filter is %s" % where_suffix)
        where_suffix = ""
        for func in self.dh.get_function_embeddings(target_db, start=start, end=end, where_suffix=where_suffix):
            target_funcs.append(func)
        return self.function_embedding_similarity(source_funcs, target_funcs, threshold=threshold, topk=topk)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("--source_db", type=str,
                    help="source sqlite db file path, for query database")
    ap.add_argument("--target_db", type=str,
                    help="target sqlite db file path, for target database")
    ap.add_argument("--result", type=str, default="./results/default.json", help="file path to save search results")
    return ap.parse_args()


def log_result(args, result):
    result_file = ""

    # if result file is specified with suffix '.json', use json.dump to save result
    if args.result.endswith(".json"):
        if not os.path.exists(os.path.dirname(args.result)):
            l.error("The Specified Output file does not exists.")
            return
        json.dump(result, open(args.result, 'w'))
        return

    if len(args.result) > 0:
        result_file = args.result
    else:
        result_file = "%s_%s.result" % (os.path.basename(args.source_db), os.path.basename(args.target_db))
    with open(result_file, 'w') as f:
        f.write("******Time: %s ******\n"%str(datetime.datetime.now()))
        f.write("******Args: %s ******\n" % str(args))
        for res in result:
            s_func_info = result[res]['info']
            f.write("[QUERYFUNC]:%20s\tQUERYELF:%20s\n" % (s_func_info["FuncName"],s_func_info["BinaryPath"]))
            if "CVE_ID" in s_func_info:
                f.write("[QUERY CVE_ID]:%20s\tCVE DESCRIPTION:%20s\tCVE REFERENCES:%20s\n" % (s_func_info["CVE_ID"], s_func_info["CVE_DESCRIPTION"], s_func_info["CVE_REFERENCES"]))
            for t_func_info in result[res]['rank']:
                f.write("\t|Func:%20s\t|ELFPath:%40s\n" %(t_func_info["FuncName"], t_func_info["BinaryPath"]))
                if "CVE_ID" in t_func_info:
                    f.write("\t[CVE_ID]:%20s\t[CVE DESCRIPTION]:%20s\t[CVE REFERENCES]:%20s\n" %  (t_func_info["CVE_ID"], t_func_info["CVE_DESCRIPTION"], t_func_info["CVE_REFERENCES"]))



def mfen_search():
    args = parse_args()
    print(args)
    application_args = ApplicationArguments()
    bs = BinarySearch(application_args)
    threshold = application_args.threshold
    topk = application_args.topk
    if threshold < -1 or threshold > 1:
        print("Threshold Value Error! range is -1~1!")
        exit(1)
    result = bs.db_similarity(args.source_db, args.target_db, threshold=threshold, topk = topk)
    log_result(args, result)

if __name__ == '__main__':
    mfen_search()