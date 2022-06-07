# encoding: utf-8

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    hidden_dropout_prob: float = field(default=0.2, metadata={"help": "pretrain model output embedding dropout"})
    inner_dim: int = field(default=512, metadata={"help": "inner dim in cls heads"})
    clr_nums: list = field(default_factory=list, metadata={"help": "classification label nums (list) for multi tasks"})
    clr_targets: list = field(default_factory=list, metadata={"help": "classification task names (list)"})
    reg_targets: list = field(default_factory=list, metadata={"help": "regression task names (list)"})
    classifier_dropout: float = field(default=0.1, metadata={"help": "classifier dropout"})

    pretrained_model: str = field(default="./checkpoints/pretrain/best", metadata={"help": "pretrained model path"})
    


@dataclass
class CustomTrainingArguments:
    checkpoint_dir: str = field(default="./checkpoints/type_4/last", metadata={"help": "saved checkpoint dir"})
    best_dir: str = field(default="./checkpoints/type_4/best", metadata={"help": "best checkpoint dir"})
    do_eval: bool = field(default=True, metadata={"help": "do eval during training"})
    epoch: int = field(default=5, metadata={"help": "epoch num"})
    lr: float = field(default=1e-4, metadata={"help": "learning rate"})
    train_batch_size: int = field(default=64, metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=64, metadata={"help": "eval batch size"})
    patience: int = field(default=1, metadata={"help": "early stop patience"})
    
    load_checkpoint: bool = field(default=False, metadata={"help": "whethe load pretrained checkpoint"})
    load_checkpoint_dir: str = field(default="./checkpoints/type_4/best", metadata={"help": "loaded pretrained checkpoint dir"})
    

@dataclass
class DataArguments:
    root: str = field(default="./data_file/split_data_file", metadata={"help": "data root path"})
    train_file: str = field(default="", metadata={"help": "train data path"})
    valid_file: str = field(default="", metadata={"help": "valid data path"})
    test_file: str = field(default="", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    limit: int = field(default=-1, metadata={"help": "max count of the loaded data"})

    read_cache: bool = field(default=True, metadata={"help": "reading func data cache or not"})
    cache_path: str = field(default="./cache/type/", metadata={"help": "the func data cache path(pickle)"})
    max_length: int = field(default=512, metadata={"help": "input sequence max length"})
    max_pos_length: int = field(default=512, metadata={"help": "position sequence max length"})

