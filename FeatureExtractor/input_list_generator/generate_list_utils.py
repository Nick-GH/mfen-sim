# coding:UTF-8
import os
import sys
import time
# print(sys.path)
from tqdm import tqdm
from utils import *
from subprocess import Popen, PIPE


def system(cmd):
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    return out.decode().strip()


def generateInputListFromFileName(elf_dir,output_dir, output_file,include_packages = [], exclude_packages = [], 
    include = False, exclude = False, compilers = [], archs = [], optis = [], bin_names = []):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # input_list = output_dir + output_type + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + "_" + output_filename
    input_list = output_dir + os.sep + output_file
    # print(input_list)
    all_files = []
    
    for root,dirs,files in os.walk(elf_dir):
        # print(root)
        files = list(map(lambda x:os.path.join(root,x),files))

        # files = list(filter(lambda x: "ELF" in str(system("file %s" % x)),files))
        files = list(filter(lambda x:x.endswith(".elf"),files))
        # print(files)
        all_files.extend(files)
        print("current %d files" % len(files))
    # print(all_files)
    all_files = list(map(lambda x:(x,parse_fname(x.split(os.sep)[-1])), all_files))
    print("total %d files" % len(all_files))
    if include:
        all_files = list(filter(lambda x:x[1] and x[1]['package'].split("-")[0] in include_packages, all_files))
        print("total %d files after including packages" % len(all_files))
    if exclude:
        all_files = list(filter(lambda x:x[1] and x[1]['package'].split("-")[0] not in exclude_packages, all_files))
        print("total %d files after excluding packages" % len(all_files))
    if compilers and len(compilers) > 0:
        print(compilers)
        all_files = list(filter(lambda x:x[1] and x[1]['compiler'] in compilers, all_files))
        print("total %d files after filtering compilers" % len(all_files))
    if archs and len(archs) > 0:
        all_files = list(filter(lambda x:x[1] and x[1]['arch'] in archs, all_files))
        print("total %d files after filtering archs" % len(all_files))
    if optis and len(optis) > 0:
        all_files = list(filter(lambda x:x[1] and x[1]['opti'] in optis, all_files))
        print("total %d files after filtering optis" % len(all_files))
    if bin_names and len(bin_names) > 0:
        all_files = list(filter(lambda x:x[1] and x[1]['bin_name'] in bin_names, all_files))
        print("total %d files after filtering binaries" % len(all_files))

    elf_paths = list(map(lambda x:x[0], all_files))
    with open(input_list,"w") as f:
        for elf in elf_paths:
            f.write(elf+os.linesep)
    print("find total %d binarys" % len(elf_paths))

def generateSourceList(source_dir,output_dir,output_file,include_packages=[], exclude_packages=[], include = False, exclude = False, git_packages = []):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir/gnu/source_list.txt
    source_list = output_dir  + os.sep + output_file

    source_paths = []

    packages = []
    for package in os.listdir(source_dir):
        packages.append(package)
    
    if include:
        packages = list(filter(lambda x:x in include_packages, packages))
    if exclude:
        packages = list(filter(lambda x:x not in exclude_packages, packages))
    # print(packages)
    git_paths = list(map(lambda x:os.sep.join([source_dir,x]),filter(lambda x:x in git_packages, packages)))

    for path in git_paths:
        print(path)
        source_paths.append(path)
    
    zip_package_paths = list(map(lambda x:os.sep.join([source_dir,x]),filter(lambda x:x not in git_packages, packages)))
    
    for package_path in zip_package_paths:
        # print(package_path)
        for package_version  in os.listdir(package_path):
            if package_version.endswith(".gz") or package_version.endswith(".xz"):
                continue
            source_path = package_path + os.sep + package_version
            source_paths.append(source_path)
    
    # print(source_paths)
    with open(source_list,"w") as f:
        for source in source_paths:
            f.write(source+os.linesep)
    print("generated source file from %s..." % source_dir)


def rmGeneratedFilesFromFileName(elf_dir, include_packages = [], exclude_packages = [], 
    include = False, exclude = False, compilers = [], archs = [], optis = [], bin_names = [],
    suffixs = [".pickle",".done",".output",".idb",".i64"]):

    del_files = []
    
    for root,dirs,files in os.walk(elf_dir):
        # print(root)
        for suffix in suffixs:
            # print(files)
            tmp_files = list(map(lambda x:os.path.join(root,x),files))
            tmp_files = list(filter(lambda x:x.endswith(suffix),tmp_files))
            del_files.extend(tmp_files)
    
    # print(del_files)
    
    del_files = list(map(lambda x:(x,parse_fname(x.split(os.sep)[-1])), del_files))
    print("total %d files" % len(del_files))
    if include:
        del_files = list(filter(lambda x:x[1] and x[1]['package'].split("-")[0] in include_packages, del_files))
        print("total %d files after including packages" % len(del_files))
    if exclude:
        del_files = list(filter(lambda x:x[1] and x[1]['package'].split("-")[0] not in exclude_packages, del_files))
        print("total %d files after excluding packages" % len(del_files))
    if compilers and  len(compilers) > 0:
        del_files = list(filter(lambda x:x[1] and x[1]['compiler'] in compilers, del_files))
        print("total %d files after filtering compilers" % len(del_files))
    if archs and len(archs) > 0:
        del_files = list(filter(lambda x:x[1] and x[1]['arch'] in archs, del_files))
        print("total %d files after filtering archs" % len(del_files))
    if optis and len(optis) > 0:
        del_files = list(filter(lambda x:x[1] and x[1]['opti'] in optis, del_files))
        print("total %d files after filtering optis" % len(del_files))
    if bin_names and len(bin_names) > 0:
        del_files = list(filter(lambda x:x[1] and x[1]['bin_name'] in bin_names, del_files))
        print("total %d files after filtering binaries" % len(del_files))

    del_paths = list(map(lambda x:x[0], del_files))
    # print(del_paths)
    for del_path in del_paths:
        os.remove(del_path)
    print("remove %d generated files" % len(del_paths))
    
            
                

