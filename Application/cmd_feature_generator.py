#encoding=utf-8
'''
python3
A script for facilitating the usage of feature_generator.py
'''
import sys, os, logging, argparse
root = os.path.dirname(os.path.abspath(__file__))
import subprocess
from ida.utils import get_file_type


from datetime import datetime, time
def parse_arg():
    argparser = argparse.ArgumentParser(description="Feature generator script based on IDA v7.5 and \033[1;31m Python3 \033[0m!")
    argparser.add_argument("--ida_path",
                           help="path to idal(before 7.2) or idat(after 7.2) with decompiler plugins, idal64 for 64bit binary(also for 32bit) for linux",
                           default="/data/gzh/IDA_Pro_v7.5/")
    argparser.add_argument("--use_wine", action="store_true",
                           help="use wine to execute windows version ida on linux platform, if enabled, the ida_path should be path to ida.exe or ida64.exe")
    argparser.add_argument("--binary", help="path to binary to be analysed", default="")
    argparser.add_argument("--directory", help="A path where all binaries in the dir will be analysed. --binary will not work if this option specified")
    argparser.add_argument("--database", help="path to sqlite database to save the extracted func data", default="default.sqlite")
    argparser.add_argument('--logfile', type=str, help="log file when ida runs", default="cmd_feature_generator.log")
    argparser.add_argument('--function', help="specific function name, default is to get all function feature", default="")
    argparser.add_argument("--compilation", choices=["default","O0","O1","O2","O3","Os"], default="default",
                           help="specify the compilation level of binary")
    argparser.add_argument("--compiler", choices=["gcc", "clang"], default="gcc",help="specify the compiler of binary")
    argparser.add_argument("--timeout", default=720, type=int, help="max seconds the compilation of a binary cost")
    argparser.add_argument("--cve_csv_path", type=str, help="the path to cve csv file where cve functions are saved, use for generating vulnerability database")

    return argparser.parse_args()

class FeatureGenerator():
    '''
    now support ELF related files
    This class implements these functions:
    1. Extract all functions feature from the specified binary file and save to the database
    2. Extract all function feature from the specified binary file and save to the database
    3. Batch version of function1: accessing all binary files from the specified folder, extract all function features and save to Database
    4. According to a csv file containing CVE information, extract specific function from binaries and save to a vulnerability Database
    '''
    def __init__(self, args):
        self.Script = os.path.join(root,'ida','funcdata_generator.py')  #the path to the IDAPython script for extracting func feature
        self.args = args
        self.logger = logging.getLogger("FeatureGenerator")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
    


    def extract_function_feature(self, binary_path): # the described function 1,2,4
        arch = get_file_type(binary_path)
        if arch is None:
            self.logger.warning("Skip Unknown file type: %s" % binary_path)
            return
        database = self.args.database
        # database = binary_path + "_" + self.args.database
        IDA_ARGS = ["-o %s" % self.args.compilation,  "-g %s" % args.compiler, "-d %s" % database]
        if self.args.function:
            IDA_ARGS.append("-f %s" % self.args.function)
        if self.args.cve_csv_path:
            IDA_ARGS.append("-cve %s" % self.args.cve_csv_path)
        IDA_ARGS = " ".join(IDA_ARGS)
        if self.args.use_wine:
            ida_path = self.args.ida_path
            if arch.find("_32") != -1:
                ida_path = os.path.join(ida_path, "ida.exe")
            elif arch.find("_64") != -1:
                ida_path = os.path.join(ida_path, "ida64.exe")
            cmd_list = ["TVHEADLESS=1", "wine", ida_path, 
                "-c" ,"-A", '-S"%s %s"' % (self.Script, IDA_ARGS)]
            
        else:
            ida_path = self.args.ida_path
            if arch.find("_32") != -1:
                ida_path = os.path.join(ida_path, "idal")
            elif arch.find("_64") != -1:
                ida_path = os.path.join(ida_path, "idal64")
            cmd_list = ["TVHEADLESS=1",  ida_path, "-c" ,"-A", '-S"%s %s"' % (self.Script, IDA_ARGS)]
        
        if self.args.logfile:
            cmd_list.append("-L{}".format(self.args.logfile))
        cmd_list.append(binary_path)
        cmd = " ".join(cmd_list)
        
        p = subprocess.Popen(cmd, shell =True)
        try:
            p.wait(timeout=self.args.timeout) # after waiting , kill the subprocess
        except subprocess.TimeoutExpired as e:
            self.logger.error("[Error] time out for %s" % binary_path)
        if p.returncode != 0:
            self.logger.error("[ERROR] cmd: %s" %(cmd))
            if p:
                p.kill()
        else:
            self.logger.info("[OK] cmd %s " % cmd)

    def extract_feature_from_dir(self, dir):# the described function 3
        from tqdm import tqdm
        for root, dirs, files in os.walk(dir):
            for f in tqdm(files, desc=os.path.basename(root)):
                binary_path = os.path.join(root, f)
                self.extract_function_feature(binary_path)

# usage:
# cmd_feature_generator.py calls funcdata_generator.py to reverse binaries and generate db.
# application.py uses model to generate embeddings of functions in db
# main_app.py computes the similarity of two dbs.

if __name__ == '__main__':
    args =parse_arg()
    # print(args)
    fg = FeatureGenerator(args)
    import time
    print(time.ctime())
    if args.directory:
        fg.extract_feature_from_dir(args.directory)
    elif args.binary:
        fg.extract_function_feature(args.binary)
    print(time.ctime())
