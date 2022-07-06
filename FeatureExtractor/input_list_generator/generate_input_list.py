import os
from generate_list_utils import *
import time
import yaml
from argparse import ArgumentParser

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--config_path", type=str, 
                    help="yml config file path for generating the input list",
                    default="config/pretrain.yml")
    args = ap.parse_args()
    config_path = args.config_path
    config = None
    with open(config_path, encoding="UTF-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(configs)
    
    force = config["force"]
    removes = config["removes"]

    source_dir = config["source"]["source_dir"]
    source_output_dir = config["source"]["output_dir"]
    source_output_file = config["source"]["output_file"]
    
    elf_dir = config["binary"]["elf_dir"]
    binary_output_dir = config["binary"]["output_dir"]
    binary_output_file = config["binary"]["output_file"]

    include_packages = config["include_packages"]
    exclude_packages = config["exclude_packages"]
    include = config["include"]
    exclude = config["exclude"]

    compilers = config["compilers"]
    archs = config["archs"]
    optis = config["optis"]
    bin_names = config["bin_names"]
    git_packages = config["git_packages"]
    
    if force:
        rmGeneratedFilesFromFileName(elf_dir = elf_dir,
            include_packages = include_packages,
            exclude_packages = exclude_packages,
            include = include,
            exclude = exclude,
            compilers = compilers,
            archs = archs,
            optis = optis,
            bin_names = bin_names,
            suffixs= removes)
    

    generateSourceList(source_dir = source_dir, 
        output_dir = source_output_dir,
        output_file = source_output_file,
        include_packages = include_packages,
        exclude_packages = exclude_packages,
        include = include,
        exclude = exclude,
        git_packages = git_packages)
    
    generateInputListFromFileName(elf_dir = elf_dir,
        output_dir= binary_output_dir,
        output_file = binary_output_file,
        include_packages = include_packages,
        exclude_packages = exclude_packages,
        include = include,
        exclude = exclude,
        compilers = compilers,
        archs = archs,
        optis = optis,
        bin_names = bin_names)