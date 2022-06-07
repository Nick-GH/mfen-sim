# coding:utf-8

# train tokenizer
import os
from data import Function, BasicBlock
from pathlib import Path
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer, Tokenizer
from tqdm import tqdm
import pickle

class TokenizerTrainer:
    def __init__(self, file, data_file="", vocab_size=20000, min_frequency=2, limit_alphabet=20000, load_data=False) -> None:
        super().__init__()
        self.file = file
        self.data_file = data_file
        self.vocab_size= vocab_size
        self.min_frequency = min_frequency
        self.limit_alphabet = limit_alphabet
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.load_data = load_data
        self.tokenzier = BertWordPieceTokenizer(clean_text=True, strip_accents=True, lowercase=False)
        pass

    def _preprocess(self):
        print("Loading func datas and generate trainer files...")
        assert os.path.exists(self.data_file)
        with open(self.data_file, "rb") as f:
            func_datas = pickle.load(f)
            print("Total loading {} functions".format(str(len(func_datas["X,X,X"]))) )
            with open("./tmp.txt", "w") as output:
                for func_data in tqdm(func_datas["X,X,X"]):
                    output.write( Function(func_data).get_func_inst_seq() + os.linesep )
        print("Generate tokenizer training files.")
        self.file = "./tmp.txt"

    
    def train_and_save_tokenizer(self):
        if self.load_data:
            self._preprocess()
        
        self.tokenzier.train(self.file, 
            vocab_size= self.vocab_size, 
            min_frequency=self.min_frequency, 
            show_progress=True,
            special_tokens=self.special_tokens,
            limit_alphabet=self.limit_alphabet,
            wordpieces_prefix="##")
        self.tokenzier.save_model("tokenizer")

if __name__ == "__main__":
    data_file = "path_to_data_pickle"
    asm_dir = "path_to_.asm"
    asm_dir = "./tmp.txt"
    tokenizer = TokenizerTrainer(file=asm_dir, data_file= data_file, load_data=False)
    tokenizer.train_and_save_tokenizer()
