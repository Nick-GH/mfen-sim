#!/usr/bin/env bash
idapath="path/to/IDA"
binary="abspath/to/binary"
vul_directory="abspath/to/vul_directory"
vul_db="abspath/to/vul_db"
cve_csv_path="abspath/to/cve_csv_file"
timeout=720


export WINEDEBUG=fixme-all,err-all
python cmd_feature_generator.py \
    --ida_path $idapath \
    --directory $vul_directory \
    --database $vul_db \
    --cve_csv_path $cve_csv_path \
    --use_wine 

echo "Encoding functions in db..."
python application.py \
    --db_path $vul_db