# encoding: utf-8

from dataclasses import dataclass, field

    


@dataclass
class TypeDataArguments:
    root: str = field(default="./cache/graph", metadata={"help": "data root path"})
    train_file: str = field(default="train_filtered_functions.pickle", metadata={"help": "train data path"})
    valid_file: str = field(default="valid_filtered_functions.pickle", metadata={"help": "valid data path"})
    test_file: str = field(default="test_filtered_functions.pickle", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    limit: int = field(default=-1, metadata={"help": "max count of the loaded data"})

    read_cache: bool = field(default=True, metadata={"help": "reading func data cache or not"})
    cache_path: str = field(default="./cache/type_old/", metadata={"help": "the func data cache path(pickle)"})
    max_length: int = field(default=512, metadata={"help": "input sequence max length"})
    max_pos_length: int = field(default=512, metadata={"help": "position sequence max length"})
    pred_arg_num: int = field(default=5, metadata={"help": "dummy argument for data loader"})

@dataclass
class CodeLiteralDataArguments:
    root: str = field(default="./cache/graph", metadata={"help": "data root path"})
    train_file: str = field(default="train_filtered_functions.pickle", metadata={"help": "train data path"})
    valid_file: str = field(default="valid_filtered_functions.pickle", metadata={"help": "valid data path"})
    test_file: str = field(default="test_filtered_functions.pickle", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    
    limit: int = field(default=-1, metadata={"help": "dataset limit"})
    read_cache: bool = field(default=True, metadata={"help": "reading the function cache or not"})
    cache_path: str = field(default="./cache/graph/", metadata={"help": "the function cache path(pickle)"})

    max_length: int = field(default=256, metadata={"help": "input sequence max length"})
    max_length_type: int = field(default=512, metadata={"help": "input sequence max length for type"})
    max_pos_length: int = field(default=512, metadata={"help": "position sequence max length"})

