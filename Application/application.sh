#!/usr/bin/env bash
idapath="/data/gzh/IDA_Pro_v7.5/"
binary="abspath/to/binary"

source_directory="abspath/to/directory"
source_db="abspath/to/binary_db"
target_db="path/to/vul_db"
timeout=720


export WINEDEBUG=fixme-all,err-all
echo "Processing IDA analysis ..."
python cmd_feature_generator.py \
    --ida_path $idapath \
    --binary $binary \
    --directory $source_directory \
    --database $source_db \
    --use_wine 


echo "Encoding functions in db..."
python application.py \
    --db_path $source_db


echo "Calculate similarity between dbs..."
python main_app.py \
    --source_db $source_db \
    --target_db $target_db \
