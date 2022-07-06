# encoding: utf-8

from dataclasses import dataclass, field


@dataclass
class ModelArguments:

    hidden_dim: int = field(default=512, metadata={"help": "hidden layer dim"}) # gat_hidden_dim % head_num == 0

    output_dim: int = field(default=1536, metadata={"help": "output dim in similarity heads"})
    gnn_layers: int = field(default=2, metadata={"help": "the layers of gat"})
    gnn_heads: int = field(default=8, metadata={"help": "the attention heads of gat"})
    use_string_and_const: bool = field(default=True, metadata={"help": "use code literal feature"})
    use_type: bool = field(default=True, metadata={"help": "use finetuned func signature prediction feature"})
    
    pretrain_model: str = field(default="./checkpoints/pretrain/best", metadata={"help": "pretrained bert model path"})
    finetune_model: str = field(default="./checkpoints/type/best", metadata={"help": "finetuned bert_cls model path"})
    use_projector: bool = field(default=True, metadata={"help": "use projector"})
    fix_pretrained_model: bool = field(default=False, metadata={"help": "fixed the pretrained model parameters"})

    clr_nums: list = field(default_factory=list, metadata={"help": "classification label nums (list) for function signature prediction tasks"})
    clr_targets: list = field(default_factory=list, metadata={"help": "classification task names (list)"})

@dataclass
class CustomTrainingArguments:
    checkpoint_dir: str = field(default="./checkpoints/similarity/sim/last/", metadata={"help": "saved checkpoint dir"})
    best_dir: str = field(default="./checkpoints/similarity/sim/best/", metadata={"help": "best checkpoint dir"})
    do_eval: bool = field(default=True, metadata={"help": "do eval during training"})
    epoch: int = field(default=3, metadata={"help": "epoch num"})
    lr: float = field(default=1e-5, metadata={"help": "learning rate"})
    train_batch_size: int = field(default=3, metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=12, metadata={"help": "eval batch size"})
    patience: int = field(default=2, metadata={"help": "early stop patience"})
    tmp: float = field(default=0.05, metadata={"help": "Temperature for softmax."})
    hard_negative_weight: float = field(default=0, metadata={"help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."})
    
    load_checkpoint: bool = field(default=True, metadata={"help": "load from a exist checkpoint"})
    load_checkpoint_path: str = field(default="./checkpoints/similarity/sim/best/similarityBest.pth", 
        metadata={"help": "path of the exist checkpoint"})
    


@dataclass
class DataArguments:
    root: str = field(default="./cache/graph", metadata={"help": "data root path"})
    train_file: str = field(default="train_filtered_functions.pickle", metadata={"help": "train data path"})
    valid_file: str = field(default="valid_filtered_functions.pickle", metadata={"help": "valid data path"})
    test_file: str = field(default="test_filtered_functions.pickle", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    
    limit: int = field(default=-1, metadata={"help": "dataset limit"})
    min_size: int = field(default=3, metadata={"help": "min graph size"})
    max_size: int = field(default=50, metadata={"help": "max graph size"})

    type_model: str = field(default="./checkpoints/type/best", metadata={"help": "finetuned type model path"})

    read_cache: bool = field(default=True, metadata={"help": "reading the function cache or not"})

    cache_path: str = field(default=".cache/graph/", metadata={"help": "the function cache path(pickle)"})
    embedding_path: str = field(default=".cache/embeddings/", metadata={"help": "the embedding cache path(pt)"})

    max_length: int = field(default=256, metadata={"help": "input sequence max length"})
    max_length_type: int = field(default=512, metadata={"help": "input sequence max length for type"})
    max_pos_length: int = field(default=512, metadata={"help": "position sequence max length"})

    arch_same: bool = field(default=True, metadata={"help": "the arch should be same"})
    compiler_same: bool = field(default=False, metadata={"help": "the compiler should be same"})
    opti_same: bool = field(default=True, metadata={"help": "the opti should be same"})

    cuda: bool = field(default=True, metadata={"help": "use cpu in sample selector"})

