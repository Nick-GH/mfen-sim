# encoding: utf-8

from dataclasses import dataclass, field



@dataclass
class DataArguments:
    root: str = field(default="./data_file/", metadata={"help": "data root path"})
    train_file: str = field(default="train_data.pickle", metadata={"help": "train data path"})
    valid_file: str = field(default="valid_data.pickle", metadata={"help": "valid data path"})
    test_file: str = field(default="test_data.pickle", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    limit: int = field(default=-1, metadata={"help": "max count of the loaded data"})
    min_size: int = field(default=3, metadata={"help": "min graph size"})
    max_size: int = field(default=50, metadata={"help": "max graph size"})

    cache_path: str = field(default="./cache/graph/", metadata={"help": "the function cache path(pickle)"})

