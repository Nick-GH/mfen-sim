# coding:utf-8
import os
import sys
import string

from hashlib import sha1
from collections import defaultdict

import time
import pprint as pp

import idautils
import idc
import idaapi
import ida_pro
import ida_nalt
import ida_bytes


# sys.path.insert(0,"path/to/feature_extractor")
sys.path.insert(0, ".")

print(sys.path)
from feature_extractor.utils import demangle, get_arch, init_idc, parse_fname, store_func_data
from normalization import normalization

printset = set(string.printable)
isprintable = lambda x: set(x).issubset(printset)

def vaule_norm(value, arch):
    minus_value = abs(to_minus_num(value, arch))
    return min(value, minus_value)

def to_minus_num(value, arch):
    if "32" in arch:
        return value - 2**32
    if "64" in arch:
        return value - 2**64
    return value

# find consts
def get_consts(start_addr, end_addr, arch):
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
                        consts.append(vaule_norm(imm_value, arch))
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


def main():

    bin_path = ida_nalt.get_input_file_path()

    with open(bin_path, "rb") as f:
        bin_hash = sha1(f.read()).hexdigest()
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

    # Parse option information
    package, compiler, arch, opti, bin_name = parse_fname(bin_path)
    if "_noinline" in bin_path:
        other_option = "noinline"
    elif "_pie" in bin_path:
        other_option = "pie"
    elif "_lto" in bin_path:
        other_option = "lto"
    else:
        other_option = "normal"

    # Prepare default information for processing
    caller_map, callee_map = get_call_graph()
    edge_map, bb_callee_map = get_bb_graph(caller_map, callee_map)

    # Now extract function information
    func_data = []
    for idx, addr in enumerate(list(idautils.Functions())):
        function = idaapi.get_func(addr)
        if (
            not function
            or function.start_ea == idaapi.BADADDR
            or function.end_ea == idaapi.BADADDR
        ):
            continue
        seg_name = get_segm_name(addr)
        # filter by segment name, only .text segment
        if seg_name != ".text":
            continue
        # IDA's default function information
        func_name = get_func_name(addr).strip()
       
        graph = idaapi.FlowChart(function, flags=idaapi.FC_PREDS)
        data = idc.get_bytes(addr, function.size()) or ""
        data_hash = sha1(data).hexdigest()
        stack_size = get_frame_size(addr)

        # Get imported callees. Note that the segment name is used because
        # idaapi.get_import_module_name() sometimes returns bad results ...
        imported_callees = []
        if func_name in callee_map:
            imported_callees = list(
                filter(lambda x: get_segm_name(x[1]) != get_segm_name(addr), callee_map[func_name])
            )

        
        # Get type information from IDA
        func_type, ret_type, args, res = get_type(addr)
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
            bb_consts = get_consts(bb.start_ea, bb.end_ea, arch)
            bb_callees = list(filter(lambda x: x[0] == bb.id, bb_callee_map[func_name]))
            # 加入了bb_ins
            # bb_ins = get_origin_ins(bb.start_ea,bb.end_ea)
            bb_ins = get_normalized_ins(bb.start_ea,bb.end_ea,arch)
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
        func_data.append(
            {
                "ida_idx": idx,
                "seg_name": seg_name,
                "name": func_name,
                "hash": data_hash,
                "size": function.size(),
                "startEA": function.start_ea,
                "endEA": function.end_ea,
                "cfg_size": graph.size,
                "img_base": img_base,
                "bin_path": bin_path,
                "bin_hash": bin_hash,
                "bin_offset": addr - img_base,
                "stack_size": stack_size,
                "package": package,
                "compiler": compiler,
                "arch": arch,
                "opti": opti,
                "others": other_option,
                "bin_name": bin_name,
                "func_type": func_type,
                "ret_type": ret_type,
                "args": args,
                "callers": caller_map[func_name],
                "callees": callee_map[func_name],
                "imported_callees": imported_callees,
                "cfg": edge_map[func_name],
                "strings": func_strings,
                "consts": func_consts,
                "type_info_parsed": res,
                "bb_data": bb_data,
            }
        )
    return func_data


init_idc()
try:
    func_data = main()
except:
    import traceback

    traceback.print_exc()
    ida_pro.qexit(1)
else:
    bin_path = ida_nalt.get_input_file_path()
    # binary_data = parse_fname(bin_path)
    binary_data = None
    print(bin_path)
    store_func_data(bin_path, func_data, binary_data)
    ida_pro.qexit(0)
