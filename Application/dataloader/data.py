# coding:utf-8
from ctypes import Array
import re
import copy
import collections

class Function:
    # func_data dict
    def __init__(self, func_data) -> None:
        self.func_name = func_data["name"]
        self.startEA = func_data["startEA"]
        self.endEA = func_data["endEA"]
        self.cfg_size = func_data["cfg_size"]
        self.img_base = func_data["img_base"]
        self.bin_path = func_data["bin_path"]
        self.bin_file_name = func_data["bin_file_name"] if "bin_file_name" in func_data else ""
        self.bin_offset = func_data["bin_offset"]
        self.stack_size = func_data["stack_size"]
        self.compiler = func_data["compiler"]
        self.arch = func_data["arch"]
        self.opti = func_data["opti"]
        self.args = func_data["args"]
        self.func_type = func_data["func_type"]
        self.callers = func_data["callers"]
        self.callees = func_data["callees"]
        self.imported_callees = func_data["imported_callees"]
        self.cfg_edge_list = func_data["cfg"]
        self.strings  = func_data["strings"]
        self.consts = func_data["consts"]
        self.abstract_args_type = None
        self.abstaract_ret_type = None
        if "abstract_args_type" in func_data:
            self.abstract_args_type = func_data["abstract_args_type"]
            self.abstaract_ret_type = func_data["abstract_ret_type"]
        self.bb_data = []
        for block_data in func_data["bb_data"]:
            self.bb_data.append(BasicBlock(block_data, self.arch))
        
        self.ground_truth = (self.bin_path, self.func_name)

        self.inst_num = 0
        for bb in self.bb_data:
            self.inst_num += bb.get_inst_num()
    
    def get_func_arg_type_dict(self):
        type2num = collections.defaultdict(int)
        for type in self.abstract_args_type:
            type2num[type] += 1
        return type2num
    
    def get_bb_insts(self, concat_tokens=False, replace_imm=False):
        ins_seq = []
        for bb in self.bb_data:
            ins_seq.append(bb.get_inst_seq(concat_tokens, replace_imm))
        
        return ins_seq
    
    def get_func_strings(self):
        if len(self.strings) == 0:
            return "Empty"
        strings = []
        for str in self.strings:
            # inst_addr, str, str_type, str_addr
            str = str[1]
            strings.append(remove_split(str))
        
        return " ".join(strings).strip()

    def get_func_consts(self):
        if len(self.consts) == 0:
            return "Empty"
        for i in range(len(self.consts)):
            self.consts[i] = vaule_norm(self.consts[i], self.arch)
        self.consts.sort()
        consts = list(map(lambda x:str(x), self.consts))
        return " ".join(consts).strip()
    
    def get_adj_bb_pair(self, concat_tokens=False, replace_imm=False):
        if self.cfg_size == 1:
            return None
        else:
            bb_pairs = []
            for src,dest in self.cfg_edge_list:
                bb_pair = (self.bb_data[src].get_inst_seq(concat_tokens,replace_imm), self.bb_data[dest].get_inst_seq(concat_tokens,replace_imm))
                bb_pairs.append(bb_pair)
            return bb_pairs
    
    def get_not_adj_bb_pair(self):
        if self.cfg_size == 1:
            pass
        else:
            pass

    def get_cfg_edge_list(self):
        if self.cfg_size == 1:
            return [ [0], [0] ]
        sources = []
        targets = []
        for src,target in self.cfg_edge_list:
            sources.append(src)
            targets.append(target)
        return [sources, targets]

    def get_cfg_adj_matrix(self):
        pass




class BasicBlock:
    def __init__(self, block_data, arch):
        self.block_id = block_data["block_id"]
        self.size = block_data["size"]
        self.startEA = block_data["startEA"]
        self.endEA = block_data["endEA"]
        self.type = block_data["type"]
        self.is_ret = block_data["is_ret"]
        self.callees = block_data["callees"]
        self.strings = block_data["strings"]
        self.consts = block_data["consts"]
        self.insts = block_data["bb_ins"]
        self.arch = arch

    def get_inst_num(self):
        return len(self.insts)
    
    def get_inst_seq(self, concat_tokens=False, replace_imm=False):
        bb_ins = copy.copy(self.insts)
        if replace_imm:
            for i in range(len(self.insts)):
                bb_tokens = self.insts[i].split()
                for j in range(len(bb_tokens)):
                    bb_tokens[j] = re.sub(r"^[0-9]+", " imm", bb_tokens[j])
                bb_ins[i] = " ".join(bb_tokens).stirp()
        
        if concat_tokens:
            for bb_in in bb_ins:
                bb_in = ("_").join(bb_in.split()).strip()
        
        return " ".join(bb_ins).strip()
    
    def get_strings(self):
        if len(self.strings) == 0:
            return "None"
        strings = []
        for str in self.strings:
            # print(str)
            # [h, s, t, ref]
            # inst_addr, str, str_type, str_addr
            str = str[1]
            strings.append(remove_split(str))
        
        return " ".join(strings).strip()

    def get_consts(self):
        if len(self.consts) == 0:
            return "None"
        for i in range(len(self.consts)):
            self.consts[i] = vaule_norm(self.consts[i], self.arch)
        # self.consts.sort()
        consts = list(map(lambda x:str(x), self.consts))
        return " ".join(consts).strip()
    

def vaule_norm(value, arch):
    minus_value = abs(to_minus_num(value, arch))
    return min(value, minus_value)

def to_minus_num(value, arch):
    if "32" in arch:
        return value - 2**32
    if "64" in arch:
        return value - 2**64
    return value

def remove_split(text):
    for ch in ["\n", "\t"]:
        text = text.replace(ch, " ")
    # return text.strip().split()
    return text
    