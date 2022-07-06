# coding:utf-8
import os
import sys
import string
import logging
from hashlib import sha1
from collections import defaultdict
import pickle
import time
import pprint as pp


import idautils
import idc
import idaapi
import ida_pro
import ida_nalt
import ida_bytes
import ida_kernwin
import ida_segment
import ida_funcs
import ida_xref
import ida_bytes
import ida_lines
import ida_ua

sys.path.insert(0, ".")

print(sys.path)

from normalization import normalization
from utils import get_arch, init_idc
from db_operation import DBOP

import csv

l = logging.getLogger("funcdata_generator.py")
logpath = os.path.join(os.path.dirname(__file__), "log")
if not os.path.exists(logpath):
    os.mkdir(logpath)
l.addHandler(logging.FileHandler(os.path.join(logpath, "datahelper.log")))
l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)

#-------------------------------------------------------
# ida reverse
printset = set(string.printable)
isprintable = lambda x: set(x).issubset(printset)

def vaule_norm(value, bits):
    minus_value = abs(to_minus_num(value, bits))
    return min(value, minus_value)

def to_minus_num(value, bits):
    if bits == 16:
        return value - 2**16
    if bits == 32:
        return value - 2**32
    if bits == 64:
        return value - 2**64
    return value

# find consts
def get_consts(start_addr, end_addr, bits):
    consts = []
    for h in idautils.Heads(start_addr, end_addr):
        insn = DecodeInstruction(h)
        if insn:
            for op in insn.ops:
                if op.type == idaapi.o_imm:
                    # get operand value
                    imm_value = op.value
                    # check if addres is loaded in idb
                    if not ida_bytes.is_loaded(imm_value):
                        consts.append(vaule_norm(imm_value, bits))
    return consts

# find strings
def get_strings(start_addr, end_addr):
    strings = []
    for h in idautils.Heads(start_addr, end_addr):
        refs = idautils.DataRefsFrom(h)
        for ref in refs:
            t = idc.get_str_type(ref)
            if isinstance(t, int) and t >= 0:
                s = idc.get_strlit_contents(ref)
                s = s.decode("utf-8")
                if s and isprintable(s):
                    strings.append([h, s, t, ref])
    return strings

# get instructions
def get_origin_ins(start_addr, end_addr):
    ins = []
    for h in idautils.Heads(start_addr, end_addr):
        cur_in = idc.GetDisasm(h)
        ins.append(cur_in)
    return ins
def get_normalized_ins(start_addr,end_addr,arch):
    ins = []
    for h in idautils.Heads(start_addr, end_addr):
        cur_in = normalization(start_addr, h, arch)
        ins.append(cur_in)
    return ins

# This function returns a caller map, and callee map for each function.
def get_call_graph():
    callee_map = defaultdict(list)
    caller_map = defaultdict(list)
        
    for callee_ea in idautils.Functions():
        # callee = idaapi.get_func(callee_ea)
        # if not callee:
        #     # print("callee_ea: ", callee_ea)
        #     continue
        callee_name = idc.get_func_name(callee_ea)

        # CodeRefsTo(callee_ea, 0)
        # @param ea:   Target address
        # @param flow: Follow normal code flow or not
        # @type  flow: Boolean (0/1, False/True)
        for caller_ea in CodeRefsTo(callee_ea, 0):
            # get_func: any address in a func
            # caller = idaapi.get_func(caller_ea)
            # if not caller:
            #     print("caller_ea: ", caller_ea)
            #     continue

            caller_name = idc.get_func_name(caller_ea)
            if not caller_name:
                continue
            # print(caller_name, callee_name)
            # print(caller_ea, callee_ea)
            callee_map[caller_name].append([callee_name, callee_ea])
            caller_map[callee_name].append([caller_name, caller_ea])

    return caller_map, callee_map


# This function returns edges, and updates caller_map, and callee_map
def get_bb_graph(caller_map, callee_map):
    edge_map = {}
    bb_callee_map = {}
    for func_ea in idautils.Functions():
        func = idaapi.get_func(func_ea)
        if not func or func.start_ea == idaapi.BADADDR or func.end_ea == idaapi.BADADDR:
            continue

        graph = idaapi.FlowChart(func, flags=idaapi.FC_PREDS)
        func_name = idc.get_func_name(func.start_ea)
        edge_map[func_name] = []
        bb_callee_map[func_name] = []
        for bb in graph:
            if bb.start_ea == idaapi.BADADDR or bb.end_ea == idaapi.BADADDR:
                continue

            for succbb in bb.succs():
                edge_map[func_name].append((bb.id, succbb.id))

            for callee_name, callee_ea in callee_map[func_name]:
                # Get address where current function calls a callee.
                if bb.start_ea <= callee_ea < bb.end_ea:
                    bb_callee_map[func_name].append((bb.id, callee_name, callee_ea))

    return edge_map, bb_callee_map


def get_type(addr):
    tif = idaapi.tinfo_t()
    res = ida_nalt.get_tinfo(tif, addr)
    funcdata = idaapi.func_type_data_t()
    tif.get_func_details(funcdata)
    func_type = idaapi.print_tinfo("", 0, 0, PRTYPE_1LINE, tif, "", "")
    ret_type = idaapi.print_tinfo("", 0, 0, PRTYPE_1LINE, funcdata.rettype, "", "")
    args = []
    for i in range(funcdata.size()):
        arg_type = idaapi.print_tinfo("", 0, 0, PRTYPE_1LINE, funcdata[i].type, "", "")
        args.append([i, funcdata[i].name, arg_type, funcdata[i].argloc.atype()])
    return [func_type, ret_type, args, res]

def read_cve_csv(csv_path):
    cve_datas = {}
    with open(csv_path, "r", encoding="gbk") as f:
        f_csv = csv.reader(f)
        col_names = next(f_csv)
        for line in f_csv:
            cve_data = {}
            for i,col_name in enumerate(col_names):
                cve_data[col_name] = line[i]
            func_names = cve_data["FunctionName"].strip().split(",")
            versions = cve_data["CollectedVersion"].strip().split(",")
            for func_name in func_names:
                for version in versions:
                    cve_datas[(func_name, cve_data["Package"]+"-"+version)] = (cve_data["Name"], cve_data["Description"], cve_data["References"])
    return cve_datas
    

#--------------------------------------------------------------------------
# 
class FuncDataGenerator():
    def __init__(self, optimization_level = "default", compiler = "gcc", cve_csv_path=None):
        '''
        :param optimization_level: the level of optimization when compile
        :param compiler: the compiler name like gcc
        '''
        if optimization_level not in ["O0","O1","O2","O3","Os","default"]:
            l.warning("No specific optimization level !!!")
        self.optimization_level =optimization_level
        self.compiler = compiler
        self.bin_path = ida_nalt.get_input_file_path() # path to binary
        self.file_name = ida_nalt.get_root_filename() # name of binary
        # get process info
        self.bits, self.arch, self.endian, self.img_base = self._get_process_info()
        self.caller_map, self.callee_map = get_call_graph()
        self.edge_map, self.bb_callee_map = get_bb_graph(self.caller_map, self.callee_map)
        self.function_info_list = list()
        self.cve_datas = None
        if cve_csv_path:
            self.cve_datas = read_cve_csv(cve_csv_path)
            l.info("Using cve information...")
            print(len(self.cve_datas))
        #Save the information of all functions, which is saved using pick.dumps
    
    def _get_process_info(self):
        '''
        :return: bit, arch, endian, img_base
        '''
        img_base = idaapi.get_imagebase()
        info = idaapi.get_inf_structure()
        if info.is_64bit():
            bits = 64
        elif info.is_32bit():
            bits = 32
        else:
            bits = 16

        endian = "little"
        if info.is_be():
            endian = "big"
        arch = "_".join([info.procName, str(bits), endian])
        arch = get_arch(arch)
        return bits, arch, endian, img_base
    
    def run(self, specical_name = ""):
        '''
        :param fn: a function to handle the functions in binary
        :param specical_name: specific function name while other functions are ignored
        :return:
        '''
        if specical_name != "":
            l.info("specific function name %s" % specical_name)
        if self.cve_datas:
            l.info("Using cve datas")
        for idx, addr in enumerate(list(idautils.Functions())):
            function = idaapi.get_func(addr)
            if (
                not function
                or function.start_ea == idaapi.BADADDR
                or function.end_ea == idaapi.BADADDR
            ):
                continue
            # IDA's default function information
            func_name = get_func_name(addr).strip()
            if len(specical_name) > 0 and specical_name != func_name:
                continue
            
            if self.cve_datas:
                package_version = ""
                for split in self.bin_path.split(os.sep):
                    if (func_name, split) not in self.cve_datas:
                        continue
                    else:
                        package_version = split
                        break
                if package_version == "":
                    continue
            # filter by segment name, only .text segment
            seg_name = get_segm_name(addr)
            if seg_name != ".text":
                continue
            
            graph = idaapi.FlowChart(function, flags=idaapi.FC_PREDS)
            data = idc.get_bytes(addr, function.size()) or ""
            data_hash = sha1(data).hexdigest()
            stack_size = get_frame_size(addr)

            # Get imported callees. Note that the segment name is used because
            # idaapi.get_import_module_name() sometimes returns bad results ...
            imported_callees = []
            if func_name in self.callee_map:
                imported_callees = list(
                    # callee ea
                    filter(lambda x: get_segm_name(x[1]) != get_segm_name(addr), self.callee_map[func_name])
                )

            
            # Get type information from IDA
            func_type, ret_type, func_args, res = get_type(addr)
            # Prepare basic block information for feature extraction
            func_strings = []
            func_consts = []
            bb_data = []
            for bb in graph:
                if bb.start_ea == idaapi.BADADDR or bb.end_ea == idaapi.BADADDR:
                    continue

                bb_size = bb.end_ea - bb.start_ea
                block_data = idc.get_bytes(bb.start_ea, bb_size) or b""
                block_data_hash = sha1(block_data).hexdigest()
                bb_strings = get_strings(bb.start_ea, bb.end_ea)
                bb_consts = get_consts(bb.start_ea, bb.end_ea, self.bits)
                bb_callees = list(filter(lambda x: x[0] == bb.id, self.bb_callee_map[func_name]))

                bb_ins = get_normalized_ins(bb.start_ea,bb.end_ea, self.arch)
                bb_data.append(
                    {
                        "size": bb_size,
                        "block_id": bb.id,
                        "startEA": bb.start_ea,
                        "endEA": bb.end_ea,
                        "type": bb.type,
                        "is_ret": idaapi.is_ret_block(bb.type),
                        "hash": block_data_hash,
                        "callees": bb_callees,
                        "strings": bb_strings,
                        "consts": bb_consts,
                        "bb_ins": bb_ins
                    }
                )
                func_strings.extend(bb_strings)
                func_consts.extend(bb_consts)
            self.function_info_list.append(
                {
                    "ida_idx": idx,
                    "seg_name": seg_name,
                    "name": func_name,
                    "hash": data_hash,
                    "size": function.size(),
                    "startEA": function.start_ea,
                    "endEA": function.end_ea,
                    "cfg_size": graph.size,
                    "img_base": self.img_base,
                    "bin_path": self.bin_path,
                    "bin_file_name": self.file_name,
                    "bin_offset": addr - self.img_base,
                    "stack_size": stack_size,
                    "compiler": self.compiler,
                    "arch": self.arch,
                    "opti": self.optimization_level,
                    "endian": self.endian,
                    "func_type": func_type,
                    "ret_type": ret_type,
                    "args": func_args,
                    "callers": self.caller_map[func_name],
                    "callees": self.callee_map[func_name],
                    "imported_callees": imported_callees,
                    "cfg": self.edge_map[func_name],
                    "strings": func_strings,
                    "consts": func_consts,
                    "bb_data": bb_data,
                }
            )
    
    def save_to(self, dbop):
        """
        :param db: DBOP instance
        :return:
        """
        N = 0
        
        for info in self.function_info_list:
            try:
                function_name = info["name"]
                bin_path = info["bin_path"]
                if not self.cve_datas:
                    dbop.insert_function(function_name, bin_path, info["bin_file_name"], pickle.dumps(info) )
                else:
                    package_version = ""
                    for split in self.bin_path.split(os.sep):
                        if (function_name, split) not in self.cve_datas:
                            continue
                        else:
                            package_version = split
                            break
                    if package_version == "":
                        continue
                    cve_id, cve_description, cve_references = self.cve_datas[(function_name, package_version)]
                    dbop.insert_vul_function(function_name, bin_path, info["bin_file_name"], pickle.dumps(info), \
                        cve_id, cve_description, cve_references)
                N+=1
            except Exception as e:
                l.error("insert operation exception when insert %s" % self.bin_path+" "+info["name"])
                l.error(e)
        l.info("%d records inserted" % N)
        return N

# 
if __name__ == "__main__":
    init_idc()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o","--optimization", default="default", help="optimization level when compilation")
    ap.add_argument("-f","--function", default="", help="extract the specific function info")
    ap.add_argument("-g","--compiler", default="gcc", help="compiler name adopted during compilation")
    ap.add_argument("-d","--database", default="default.sqlite", type=str, help="path to database")
    ap.add_argument("-cve", "--cve_csv_path", type=str, help="the path to cve csv file where cve functions are saved")
    args = ap.parse_args(idc.ARGV[1:])
    funcg = FuncDataGenerator(args.optimization, compiler=args.compiler, cve_csv_path=args.cve_csv_path)
    funcg.run(specical_name=args.function)
    dbop = DBOP(args.database)
    funcg.save_to(dbop)
    del dbop # free to call dbop.__del__() , flush database
    ida_pro.qexit(0)