import os
import sys
import re
import string
import random
import hashlib
import itertools
from hashlib import sha1
from subprocess import Popen, PIPE
from statistics import mean as stat_mean

import multiprocessing
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from contextlib import closing

try:
    # For old IDA versions using python2
    import cPickle as pickle
except:
    import pickle

import logging

logger = logging.getLogger(__name__)


RESTR = (
    "(.*)_"
    + "(gcc-.*|clang-.*|"
    + "gcc|clang)_"
    + "(x86_32|x86_64|arm_32|arm_64|mips_32|mips_64|mipseb_32|mipseb_64)_"
    + "(O0|O1|O2|O3|Os)_"
    + "(.*)"
)

special_package = {"libcurl": "curl"}

def transform_package_name(package):
    package = package.split("-")[0]
    if package in special_package:
        package = special_package[package]
    return package

# matches => package, compiler, arch, opti, bin_name
def parse_fname(bin_path):
    base_name = os.path.basename(bin_path)
    matches = re.search(RESTR, base_name).groups()
    return matches



def parse_source_path(src_path, package=""):
    # try:
    #     matches = re.search(RESTR, src_path)
    #     src_file = matches.groups()[-1]
    #     src_file = src_file[src_file.index('/')+1:]
        
        
    # except Exception as e:
    #     # print(e)
    #     if package == "":
    #         return ""
    #     package = transform_package_name(package)
    #     groups = src_path.split("/")
    #     src_file = ""
    #     for i,group in enumerate(groups):
    #         if package in group:
    #             src_file = "/".join(groups[i+1:])
    #             break
    matches = re.search(RESTR, src_path)
    if not matches:
        return ""
    src_file = matches.groups()[-1]
    src_file = src_file[src_file.index('/')+1:]
    # print(src_path, src_file)
    return os.path.relpath(src_file)
        

        
    

# statistics mean function cannot handle the empty list
def mean(l):
    return stat_mean(l or [0])


def flatten(l):
    return list(itertools.chain.from_iterable(l))


def system(cmd):
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    return out.decode().strip()


def randstr(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def gettmpdir():
    tmpdir = os.path.join("/tmp", "cache", randstr(10))
    while os.path.exists(tmpdir):
        tmpdir = os.path.join("/tmp", "cache", randstr(10))
    os.makedirs(tmpdir, exist_ok=True)
    return tmpdir


# For later use of custom decoding.
def decode(x):
    return x


def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."


def check_content_dup(fname, data):
    with open(fname, "rb") as f:
        f_data = f.read()
    return sha1(f_data).hexdigest() == sha1(data).hexdigest()


# Simple demangle function using 'c++filt'. IDA's internal demangler sometimes
# return bad results.
def demangle(name):
    demangled = system('c++filt -p -i "{0}"'.format(name))
    idx = 0
    b_cnt = 0
    name = []
    while idx < len(demangled):
        if demangled[idx] == "<":
            b_cnt += 1
        elif demangled[idx] == ">":
            b_cnt -= 1
        elif b_cnt == 0:
            if demangled[idx] == ":":
                name = []
            else:
                name.append(demangled[idx])
        idx += 1
    return "".join(name), demangled


def get_cache_fname(fname="", cache_dir=".cache"):
    if not fname:
        fname = randstr(10)
    if not cache_dir:
        cache_dir = os.path.join("./", ".cache")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # fname = hashlib.sha1(str(fname).encode()).hexdigest()
    cache_fname = os.path.join(cache_dir, "{}.pickle".format(fname))
    return cache_fname


def load_cache(fname="", cache_dir=".cache"):
    cache_fname = get_cache_fname(fname=fname, cache_dir=cache_dir)
    print(cache_fname)
    if not os.path.exists(cache_fname):
        return

    logger.debug("[+] Using cache file: %s" % (cache_fname))
    with open(cache_fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_cache(data, fname="", cache_dir=".cache"):
    cache_fname = get_cache_fname(fname=fname, cache_dir=cache_dir)
    logger.debug("[+] Creating cache file: %s" % (cache_fname))
    with open(cache_fname, "wb") as f:
        pickle.dump(data, f)


def system_with_cache(cmd):
    data = load_cache(fname=cmd)
    if not data:
        data = system(cmd)
        store_cache(data, fname=cmd)
    return data


def get_bytes(fname, offset, size):
    if not os.path.exists(fname):
        print(fname+" not found")
        return
    with open(fname, "rb") as f:
        f.seek(offset)
        return f.read(size)


def get_file_type(fname, use_str=False):
    if use_str:
        s = fname
    else:
        fname = os.path.realpath(fname)
        s = system("file {0}".format(fname))
    s = system('file "{0}"'.format(fname))

    if "32-bit" in s:
        bits = "32"
    elif "64-bit" in s:
        bits = "64"
    else:
        bits = None

    if "Intel 80386" or "x86" in s:
        arch = "x86"
    elif "ARM" in s or "arm" in s:
        arch = "arm"
    elif "MIPS" in s or "MIPS" in s:
        arch = "mips"
    else:
        arch = None

    if "LSB" in s:
        endian = ""
    elif "MSB" in s:
        endian = "eb"
    else:
        # endian = None
        endian = ""

    if bits is None or arch is None or endian is None:
        return None
    return "{0}{1}_{2}".format(arch, endian, bits)


# This makes IDA's architecture string compatible with ours.
def get_arch(arch):
    ret_arch = None
    if arch == "ARM_32_little":
        ret_arch = "arm_32"
    elif arch == "ARM_64_little":
        ret_arch = "arm_64"
    elif arch == "mipsl_32_little":
        ret_arch = "mips_32"
    elif arch == "mipsl_64_little":
        ret_arch = "mips_64"
    elif arch == "mipsb_32_big":
        ret_arch = "mipseb_32"
    elif arch == "mipsb_64_big":
        ret_arch = "mipseb_64"
    elif arch == "metapc_64_little":
        ret_arch = "x86_64"
    elif arch == "metapc_32_little":
        ret_arch = "x86_32"
    elif arch in [
        "arm_32",
        "arm_64",
        "mips_32",
        "mips_64",
        "mipseb_32",
        "mipseb_64",
        "x86_32",
        "x86_64",
    ]:
        ret_arch = arch
    else:
        logger.warn("Unknown architecture: %s" % (arch))
        raise NotImpelemented
    return ret_arch


def timeout_wrapper(func, *args, **kwargs):
    timeout = kwargs.get("timeout", None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
    except multiprocessing.TimeoutError:
        p.terminate()
    else:
        return out


def do_multiprocess(
    func,
    args,
    chunk_size=0,
    pool_size=8,
    # pool_size=cpu_count(),
    taskset=True,
    initializer=None,
    initargs=None,
    timeout=0,
    threshold=30000,
    force=False,
):
    if timeout > 0:
        func = partial(timeout_wrapper, func, timeout=timeout)
    if force or len(args) > threshold:
        if chunk_size == 0:
            chunk_size = len(args) // pool_size + 1
        logger.debug(
            ("[+] multiprocessing with " "{0} pool and {1} chunk size ...").format(
                pool_size, chunk_size
            )
        )
        if taskset:
            os.system("taskset -cp 0-%d %s > /dev/null" % (pool_size, os.getpid()))
        with closing(
            Pool(initializer=initializer, initargs=initargs, processes=pool_size)
        ) as pool:
            data = list(pool.imap_unordered(func, args, chunk_size))
    else:
        logger.debug("[+] no need to do multiprocessing because data is small.")
        data = []
        if initializer:
            if initargs:
                initializer(*initargs)
            else:
                initializer()
        for idx, arg in enumerate(args):
            result = func(arg)
            if result:
                data.append(func(arg))
    return data


# Belows are utilitiy functions for IDAPython
def load_plugins():
    import idaapi

    plugins_dir = idaapi.idadir("plugins")
    files = [f for f in os.listdir(plugins_dir) if re.match(r".*\.py", f)]
    for path in files:
        idaapi.load_plugin(path)


def wait_auto_analysis():
    import ida_auto

    try:
        # >= IDA Pro 7.4
        ida_auto.auto_wait()
    except AttributeError:
        # < IDA Pro 7.4
        ida_auto.autoWait()


def init_idc():
    load_plugins()
    wait_auto_analysis()


# Belows are functions for processing function data
def load_func_data(bin_path, binary_config = None, output_path = "/data/gzh/func_data", suffix=""):
    if not binary_config:
        data_path = bin_path + suffix + ".pickle"
        if not os.path.exists(data_path):
            return None,None
        with open(data_path, "rb") as f:
            func_data_list = pickle.load(f)
        return bin_path, func_data_list
    package, compiler, arch, opti, bin_name = binary_config
    package = package.split("-")[0]
    bin_name = bin_path.split(os.sep)[-1]
    bin_name = bin_name + suffix + ".pickle"
    data_path = os.sep.join([output_path, package, bin_name])
    if not os.path.exists(data_path):
        return None,None
    with open(data_path, "rb") as f:
        func_data_list = pickle.load(f)
    return bin_path, func_data_list


def store_func_data(bin_path, func_data_list, binary_data = None, output_path = "", suffix=""):
    if not binary_data:
        data_path = bin_path + suffix + ".pickle"
        with open(data_path, "wb") as f:
            pickle.dump(func_data_list, f)
        return 
    package, compiler, arch, opti, bin_name = binary_data
    package = package.split("-")[0]
    bin_name = bin_path.split(os.sep)[-1]
    bin_name = bin_name + suffix + ".pickle"
    data_path = os.sep.join([output_path, package])
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_path = os.sep.join([data_path, bin_name])
    with open(data_path, "wb") as f:
        pickle.dump(func_data_list, f)


def store_func_data_wrapper(args):
    store_func_data(*args)
