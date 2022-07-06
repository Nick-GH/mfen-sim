# coding:utf-8

import os
import sys
import string
import time
import pprint as pp
import re
from idautils import *
from idaapi import *
from idc import *
import idautils
import ida_bytes
import binascii
import ida_funcs

printset = set(string.printable)
isprintable = lambda x: set(x).issubset(printset)

num_norm_marign = 500

arm_pseudo_ins = "DCB DCW DCD DCQ".split(" ")

register_list = list(idautils.GetRegisterList())

rename_register_dic = {}
def dump_regvars(pfn):
    "dump renamed registers information"
    assert ida_funcs.is_func_entry(pfn)
    # print("Function has %d renamed registers" % pfn.regvarqty)
    for rv in pfn.regvars:
        if not rv:
            break
        rename_register_dic[rv.user] = rv.canon


# find strings
def is_string(inst_addr):
    refs = idautils.DataRefsFrom(inst_addr)
    for ref in refs:
        t = idc.get_str_type(ref)
        # print(ref, t)
        if isinstance(t, int) and t >= 0:
            s = idc.get_strlit_contents(ref)
            s = s.decode("utf-8")
            # print(s)
            if s and isprintable(s):
                # print("string "+s)
                return True
    return False

def vaule_norm(value, arch):
    minus_value = abs(to_minus_num(value, arch))
    return min(value, minus_value)

def num_norm(value, arch):
    if abs(value) > num_norm_marign:
        minus_value = to_minus_num(value, arch)
        if abs(minus_value) > num_norm_marign:
            return "imm"
        else:
            return str(minus_value)
    else:
        return str(value)

def to_minus_num(value, arch):
    if "32" in arch:
        return value - 2**32
    if "64" in arch:
        return value - 2**64
    return value


def get_register_name(operand):
    if str2reg(operand) == -1:
        if operand not in rename_register_dic:
            return "regvar"
        else:
            return rename_register_dic[operand]
    else:
        return operand


def x86_normalization(func_start_addr, inst_addr, arch):
    operator = idc.print_insn_mnem(inst_addr).lower()
    operands_type = []
    operands = []
    if operator == "":
        return "", ""
    insn = DecodeInstruction(inst_addr)
    if not insn:
        return "", ""
    ops = insn.ops
    for offset in [0, 1, 2]:
        try:
            type, reg, value, addr, phrase = ops[offset].type, ops[offset].reg, ops[offset].value, ops[offset].addr, ops[offset].phrase
            bits = ida_ua.get_dtype_size(ops[offset].dtype) * 4
            opType = get_operand_type(inst_addr, offset)
            operands_type.append(str(opType))
            if opType == o_void:
                break
            elif opType == o_far:
                seg_name = get_segm_name(addr)
                if operator.startswith("j") or operator.startswith("J"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                else:
                    operands.append("farfunc")
            elif opType == o_near:
                seg_name = get_segm_name(addr)
                if operator.startswith("j") or operator.startswith("J"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                elif addr == func_start_addr:
                    operands.append("self")
                else:
                    operands.append("nearfunc")
            elif opType == o_reg:
                operand = print_operand(inst_addr, offset)
                operands.append(get_register_name(operand))

            elif opType == o_mem:
                seg_name = get_segm_name(addr)
                operand = print_operand(inst_addr, offset)
                seg = ""
                if ":" in operand:
                    seg = operand[operand.find(":")-2: operand.find(":")]
                    seg += " : "
                if operator == "call":
                    if seg_name == ".text":
                        operands.append("innerfunc")
                    elif seg_name == ".plt" or seg_name == ".got":
                        operands.append("externfunc")
                    elif addr == func_start_addr:
                        operands.append("self")
                    else:
                        operands.append("nearfunc")
                elif seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                else:
                    operands.append(seg + "mem")
            elif opType == o_phrase or opType == o_displ:
                
                operand = print_operand(inst_addr, offset)
                mem = operand[operand.find("[")+1: operand.find("]")]
                mems = re.split("\+|-", mem)
                count = 0
                for i in range(len(mems)):
                    name = get_register_name(mems[i])
                    if name != "regvar":
                        mems[i] = name
                        count += 1
                
                ptr, scale, displ, seg = "", "", "", ""
                if operand.find("ptr") != -1:
                    i = operand.find("ptr")
                    if operand[i-1] == " ":
                        ptr_type = operand[:i-1]
                        ptr = ptr_type+"ptr"

                if "*" in operand:
                    scale = operand[operand.find("*")+1]
                    if not scale.isdigit():
                        scale = ""
                if ":" in operand:
                    index = operand.find(":")
                    seg = operand[index-2: index]
                    seg += " :"

                if opType == o_displ:
                    if is_string(inst_addr):
                        displ = "dispstr"
                    else:
                        displ = num_norm(addr, arch)
                if displ != "":
                    count += 1
                base_reg = get_register_name(mems[0])
                if count >= 2:
                    phrase_reg = get_register_name(mems[1])
                if count == 1:
                    if opType == o_phrase:
                        norm_operand = "{} {} [ {} * {} ]".format(ptr, seg, base_reg, scale) if scale \
                            else "{} {} [ {} ]".format(ptr, seg, base_reg)
                    else:
                        norm_operand = "{} {} [ {} * {} + {} ]".format(ptr, seg, base_reg, scale, displ) if scale \
                            else "{} {} [ {} + {} ]".format(ptr, seg, base_reg, displ)
                    operands.append(norm_operand)
                elif count == 2:
                    if opType == o_phrase:
                        norm_operand = "{} {} [ {} + {} * {} ]".format(ptr, seg, base_reg, phrase_reg, scale) if scale \
                            else "{} {} [ {} + {} ]".format(ptr, seg, base_reg, phrase_reg)
                    else:
                        norm_operand = "{} {} [ {} * {} + {} ]".format(ptr, seg, phrase_reg, scale, displ) if scale \
                            else "{} {} [ {} + {} ]".format(ptr, seg, base_reg, displ)
                    operands.append(norm_operand)
                elif count == 3:
                    norm_operand = "{} {} [ {} + {} * {} + {} ]".format(ptr, seg, base_reg, phrase_reg, scale, displ) if scale \
                        else "{} {} [ {} + {} + {} ]".format(ptr, seg, base_reg, phrase_reg, displ)
                    operands.append(norm_operand)
                else:
                    operands.append("dispmem")
            elif opType == o_imm:
                imm = print_operand(inst_addr, offset)

                if "-" in imm:
                    imm = imm[imm.find("(") + 1: imm.find(")")]
                    _, num = imm.split("-")
                    num = num.strip(" ")
                    num = num.strip("H")
                    num = num.strip("h")
                    if num.lower().startswith("0x") or num.isdigit():
                        num = int(num, 16)
                        value = value + num
                seg_name = get_segm_name(value)
                if seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                elif seg_name == ".text":
                    operands.append("innerfunc")
                else:
                    if imm.upper().startswith("0F") or imm.upper().startswith("0XF"):
                        value = to_minus_num(value, arch)
                    operands.append(num_norm(value, arch))
            elif opType == o_trreg:
                operands.append("tracereg")
            elif opType == o_dbreg:
                operands.append("debugreg")
            elif opType == o_crreg:
                operands.append("controlreg")
            elif opType == o_fpreg:
                operands.append("fpreg")
            elif opType == o_mmxreg:
                operands.append("mmxreg")
            elif opType == o_xmmreg:
                operands.append("xmmreg")
            else:
                operands.append("tag")
        except:
            pass
    res = operator + " " + " ".join(operands)
    return res


def arm_normalization(func_start_addr, inst_addr, arch):
    operator = idc.print_insn_mnem(inst_addr).upper() 
    operands_type = []
    operands = []
    if operator == "":
        return "", ""
    insn = DecodeInstruction(inst_addr)
    if not insn:
        return "", ""
    ops = insn.ops
    for offset in [0, 1, 2]:
        try:
            type, reg, value, addr, phrase = ops[offset].type, ops[offset].reg, ops[offset].value, ops[offset].addr, \
                                             ops[offset].phrase
            bits = ida_ua.get_dtype_size(ops[offset].dtype) * 4
            opType = get_operand_type(inst_addr, offset)
            operands_type.append(str(opType))
            if opType == o_void:
                break
                # operands.append("<VOID>")
            elif opType == o_far:
                seg_name = get_segm_name(addr)
                if operator.startswith("b") or operator.startswith("B"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                else:
                    operands.append("farfunc")
            elif opType == o_near:
                seg_name = get_segm_name(addr)
                if operator.startswith("b") or operator.startswith("B"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                elif addr == func_start_addr:
                    operands.append("self")
                else:
                    operands.append("nearfunc")
            elif opType == o_reg:  # type = 1
                operand = print_operand(inst_addr, offset)
                operands.append(get_register_name(operand))
            # type = 2
            elif opType == o_mem:
                seg_name = get_segm_name(addr)
                operand = print_operand(inst_addr, offset)
                if operator.startswith("B"):
                    if seg_name == ".text":
                        operands.append("innerfunc")
                    elif seg_name == ".plt" or seg_name == ".got":
                        operands.append("externfunc")
                    elif addr == func_start_addr:
                        operands.append("self")
                    else:
                        operands.append("nearfunc")
                elif seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                else:
                    operands.append("mem")
            elif opType == o_phrase:  
                operand = print_operand(inst_addr, offset)
                last = "!" if operand.endswith("!") else ""
                phrase_register = register_list[phrase] 
                mem = operand[operand.find("[")+1: operand.find("]")]
                registers = mem.spilt(",")
                registers[0] = phrase_register
                registers[1] = registers[1]
                mem = " , ".join(registers)
                operands.append("[ {} ] {}".format(mem, last))
            elif opType == o_displ:
                operand = print_operand(inst_addr, offset)
                if "-" in operand:
                    operand = operand[operand.find("(")+1, operand.find(")")]
                    _, num = operand.split("-")
                    num = num.strip(" ")
                    num = num.strip("H")
                    num = num.strip("h")
                    if num.lower().startswith("0x") or num.isdigit():
                        num = int(num, 16)
                        addr += num
                displ = "dispstr" if is_string(inst_addr) else num_norm(addr, arch)
                last = "!" if operand.endswith("!") else ""
                phrase_register = register_list[phrase]
                if "," in operand:
                    operands.append("[ {} , {} ] {}".format(phrase_register, displ, last))
                else:
                    operands.append("[ {} ] {}".format(phrase_register, last))
            elif opType == o_imm:
                imm = print_operand(inst_addr, offset)
                imm = imm.strip("#")
                if "-" in imm:
                    imm = imm[imm.find("(") + 1: imm.find(")")]
                    _, num = imm.split("-")
                    num = num.strip(" ")
                    num = num.strip("H")
                    num = num.strip("h")
                    if num.lower().startswith("0x") or num.isdigit():
                        num = int(num, 16)
                        value = value + num
                seg_name = get_segm_name(value)
                if seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                elif seg_name == ".text":
                    operands.append("innerfunc")
                else:
                    operands.append(num_norm(value, arch))
            elif opType == o_reglist:
                operands.append("reglist")
            elif opType == o_creglist:
                operands.append("creglist")
            elif opType == o_creg:
                operands.append("creg")
            elif opType == o_fpreglist:
                operands.append("fpreglist")
            elif opType == o_text:
                operands.append("text")
            elif opType == o_cond:
                operands.append("cond")
            else:
                operands.append("tag")
        except:
            pass
    res = operator + " " + " ".join(operands)
    return res

def mips_normalization(func_start_addr, inst_addr, arch):
    operator = idc.print_insn_mnem(inst_addr).lower() 
    operands_type = []
    operands = []
    if operator == "":
        return "", ""
    insn = DecodeInstruction(inst_addr)
    if not insn:
        return "", ""
    ops = insn.ops
    for offset in [0, 1, 2]:
        try:
            type, reg, value, addr, phrase = ops[offset].type, ops[offset].reg, ops[offset].value, ops[offset].addr, \
                                             ops[offset].phrase
            bits = ida_ua.get_dtype_size(ops[offset].dtype) * 4
            opType = get_operand_type(inst_addr, offset)
            operands_type.append(str(opType))
            if opType == o_void:
                break
            elif opType == o_far:
                seg_name = get_segm_name(addr)
                if operator.startswith("j") or operator.startswith("J"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                else:
                    operands.append("farfunc")
            elif opType == o_near:
                seg_name = get_segm_name(addr)
                if operator.startswith("j") or operator.startswith("J"):
                    operands.append("jumpdst")
                    continue
                if seg_name == ".text":
                    operands.append("innerfunc")
                elif seg_name == ".plt" or seg_name == ".got":
                    operands.append("externfunc")
                elif addr == func_start_addr:
                    operands.append("self")
                else:
                    operands.append("nearfunc")
            elif opType == o_reg:
                operand = print_operand(inst_addr, offset)
                operands.append(get_register_name(operand))
            elif opType == o_mem:
                seg_name = get_segm_name(addr)
                operand = print_operand(inst_addr, offset)
                # bal,jal
                if operator == "bal" or operator == "jal":
                    if seg_name == ".text":
                        operands.append("innerfunc")
                    elif seg_name == ".plt" or seg_name == ".got":
                        operands.append("externfunc")
                    elif addr == func_start_addr:
                        operands.append("self")
                    else:
                        operands.append("nearfunc")
                elif seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                else:
                    operands.append("mem")
            elif opType == o_phrase or opType == o_displ:  
                operand = print_operand(inst_addr, offset)
                displ = "dispstr" if is_string(inst_addr) else num_norm(addr, arch)
                phrase_register = register_list[phrase]  
                operands.append(" {} ( {} )".format(displ, phrase_register))
            elif opType == o_imm:
                imm = print_operand(inst_addr, offset)
                if "-" in imm:
                    imm = imm[imm.find("(")+1: imm.find(")")]
                    _, num = imm.split("-")
                    num = num.strip(" ")
                    num = num.strip("H")
                    num = num.strip("h")
                    if num.lower().startswith("0x") or num.isdigit():
                        num = int(num, 16)
                        value = value + num
                seg_name = get_segm_name(value)
                if seg_name == ".bss":
                    operands.append("dispbss")
                elif seg_name == ".rodata":
                    if is_string(inst_addr):
                        operands.append("dispstr")
                    else:
                        operands.append("dispdata")
                elif seg_name == ".data":
                    operands.append("dispdata")
                elif seg_name == ".data.rel.ro":
                    operands.append("dispdata")
                elif seg_name == ".text":
                    operands.append("innerfunc")
                else:
                    if imm.upper().startswith("0F") or imm.upper().startswith("0XF"):
                        value = to_minus_num(value, arch)
                    operands.append(num_norm(value, arch))
            else:
                operands.append("tag")
        except:
            pass
    res = operator + " " + " ".join(operands)
    return res

def normalization(func_start_addr, inst_addr, arch):
    pfn = ida_funcs.get_fchunk(func_start_addr)
    if pfn is None:
        print("No function at %08X!" % func_start_addr)
        return
    if ida_funcs.is_func_entry(pfn):
        dump_regvars(pfn)
    if "arm" in arch:
        return arm_normalization(func_start_addr, inst_addr, arch)
    if "x86" in arch:
        return x86_normalization(func_start_addr, inst_addr, arch)
    if "mips" in arch:
        return mips_normalization(func_start_addr, inst_addr, arch)