import os
import sys
import re


RESTR = (
    "(.*)_"
    + "(gcc-.*|clang-.*|"
    + "gcc|clang)_"
    + "(x86_32|x86_64|arm_32|arm_64|mips_32|mips_64|mipseb_32|mipseb_64)_"
    + "(O0|O1|O2|O3|Os)_"
    + "(.*)"
)

# matches => package, compiler, arch, opti, bin_name
def parse_fname(bin_path):
    compile_info ={}
    try:
        base_name = os.path.basename(bin_path)
        matches = re.search(RESTR, base_name).groups()
        compile_info["package"] = matches[0]
        compile_info["compiler"] = matches[1]
        compile_info["arch"] = matches[2]
        compile_info["opti"] = matches[3]
        compile_info["bin_name"] = matches[4]
        
    except Exception as e:
        print(e)
        return None
    
    return compile_info


