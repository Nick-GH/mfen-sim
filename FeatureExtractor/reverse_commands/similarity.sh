#!/usr/bin/env bash
idapypath="helper"
idapath=""
idc="feature_extractor/ida/fetch_funcdata_v7.5.py"
input_list=""
source_list=""
ctags_dir="data/ctags"
echo "Processing IDA analysis ..."
export WINEDEBUG=fixme-all,err-all
python $idapypath/do_idascript.py \
    --idapath $idapath \
    --idc $idc \
    --input_list $input_list \
    --log --force


