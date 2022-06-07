# encoding: utf-8

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "pretrain model output embedding dropout"})
    hidden_size: int = field(default=512, metadata={"help": "Dimensionality of the encoder layers and the pooler layer, multiple of heads"})
    num_attention_heads: int = field(default=8, metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder."})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of hidden layers in the Transformer encoder."})
    max_position_embeddings: int = field(default=512, metadata={"help": "The maximum sequence length that this model might ever be used with.\
         Typically set this to something large just in case (e.g., 512 or 1024 or 2048)."})
    intermediate_size: int = field(default=2048, metadata={"help": "Dimensionality of the 'intermediate' (often named feed-forward) layer in the Transformer encoder."})

@dataclass
class CustomTrainingArguments:
    checkpoint_dir: str = field(default="./checkpoints/pretrain/last", metadata={"help": "saved checkpoint dir"})
    best_dir: str = field(default="./checkpoints/pretrain/best", metadata={"help": "best checkpoint dir"})
    do_eval: bool = field(default=True, metadata={"help": "do eval during training"})
    epoch: int = field(default=2, metadata={"help": "epoch num"})
    lr: float = field(default=5e-5, metadata={"help": "learning rate"})
    train_batch_size: int = field(default=256, metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=256, metadata={"help": "eval batch size"})
    patience: int = field(default=1, metadata={"help": "early stop patience"})

    load_checkpoint: bool = field(default=True, metadata={"help": "whethe load pretrained checkpoint"})
    load_checkpoint_dir: str = field(default="checkpoints/pretrain/best/", metadata={"help": "loaded pretrained checkpoint dir"})


@dataclass
class DataArguments:
    root: str = field(default="./data_file", metadata={"help": "data root path"})
    train_file: str = field(default="train_data.pickle", metadata={"help": "train data path"})
    dev_file: str = field(default="valid_data.pickle", metadata={"help": "eval data path"})
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})
    mask_prob: float = field(default=0.15, metadata={"help": "mask token probability"})
    random_token_prob: float = field(default=0.1, metadata={"help": "replace with random token probability"})
    leave_unmasked_prob: float = field(default=0.1, metadata={"help": "leave unchanged probability"})
    limit: int = field(default=-1, metadata={"help": "dataset limit"})

    read_cache: bool = field(default=True, metadata={"help": "reading the bb pair cache or not"})
    cache_path: str = field(default="./cache/pretrain/", metadata={"help": "the bb pair cache path(pickle)"})
    max_length: int = field(default=256, metadata={"help": "input sequence max length"})
    max_pos_length: int = field(default=256, metadata={"help": "position sequence max length"})

