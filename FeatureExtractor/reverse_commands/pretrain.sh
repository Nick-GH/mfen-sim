#!/usr/bin/env bash
idapypath="helper"
idapath=""
idc="feature_extractor/ida/fetch_funcdata_v7.5.py"
input_list=""
ctags_dir="data/ctags"


export WINEDEBUG=fixme-all,err-all
echo "Processing IDA analysis ..."
python $idapypath/do_idascript.py \
    --idapath $idapath \
    --idc $idc \
    --input_list $input_list \
    --log --force



echo "Extract functype ..."
python ${idapypath}/extract_functype.py \
    --source_list "${source_list}" \
    --input_list "${input_list}" \
    --ctags_dir "${ctags_dir}" \
    --threshold 1
