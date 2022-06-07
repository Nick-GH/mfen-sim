# encoding: utf-8

from dataclasses import dataclass, field



@dataclass
class ApplicationArguments:
    threshold: float = field(default=-1, metadata={"help": "threshold for similarity prediction"})
    topk: int = field(default=10, metadata={"help": "display top k results"})
    cuda: bool = field(default=True, metadata={"help": "whether use cuda for similarity"})

    hidden_dim: int = field(default=512, metadata={"help": "hidden layer dim"})
    output_dim: int = field(default=1536, metadata={"help": "output dim in similarity heads"})
    gnn_layers: int = field(default=2, metadata={"help": "the layers of gcn"})
    gnn_heads: int = field(default=8, metadata={"help": "the heads of gcn"})
    use_string_and_const: bool = field(default=True, metadata={"help": "use func string and const feature"})
    use_type: bool = field(default=True, metadata={"help": "use finetuned type model"})

    pretrain_model: str = field(default="./checkpoints/pretrain/best", metadata={"help": "pretrained model path"})
    finetune_model: str = field(default="./checkpoints/type/best", metadata={"help": "finetuned type model path"})

    checkpoint_path: str = field(default="./checkpoints/similarity/best/similarityBest.pth", metadata={"help": "path of the exist checkpoint"})

    clr_nums: list = field(default_factory=list, metadata={"help": "classification label nums (list) for multi tasks"})
    clr_targets: list = field(default_factory=list, metadata={"help": "classification task names (list)"})
    
    vocab_file: str = field(default="./vocab/Bert-vocab.txt", metadata={"help": "vocab file path"})

    min_size: int = field(default=1, metadata={"help": "min graph size"})
    max_size: int = field(default=150, metadata={"help": "max graph size"})

    max_length: int = field(default=256, metadata={"help": "input sequence max length"})
    max_length_type: int = field(default=512, metadata={"help": "input sequence max length for type input"})
    max_pos_length: int = field(default=512, metadata={"help": "position sequence max length"})
