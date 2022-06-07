# coding:utf-8
# mlm, nsp and classification

from transformers import BertConfig, BertForPreTraining, BertModel
import torch
from torch import nn
from torch_geometric.nn import GATConv, GAT
from .bert_cls import BertForClassification
import torch.nn.functional as F
from torch_scatter import scatter


class MFEN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pretrained_bert = BertModel.from_pretrained(self.config.pretrain_model)
        print("BERT Loaded.")
        if config.fix_pretrained_model:
            for p in self.pretrained_bert.parameters():
                p.requires_grad = False
    
        self.gnn_embedding_dim = self.pretrained_bert.config.hidden_size
        self.final_embedding_dim = self.config.hidden_dim
        self.linear1 = None
        self.linear2 = None
  
        if self.config.use_string_and_const:
            self.code_bert_config = BertConfig.from_pretrained("microsoft/codebert-base")
            self.final_embedding_dim += self.config.hidden_dim
            self.linear1 = nn.Linear(self.code_bert_config.hidden_size, self.config.hidden_dim)
            
        
        if self.config.use_type:
            self.finetuned_bert_config = BertConfig.from_pretrained(self.config.finetune_model)
       
            self.final_embedding_dim += self.config.hidden_dim

            self.linear2 = nn.Linear(sum(self.finetuned_bert_config.clr_nums), self.config.hidden_dim)
        
        
        
        self.gnn = GAT(self.gnn_embedding_dim, self.config.hidden_dim, self.config.gnn_layers, self.config.hidden_dim, heads=self.config.gnn_heads)
        
        if self.config.use_projector:
            self.embedding_head = EmbeddingHead(self.final_embedding_dim, self.config.output_dim, activation_fn="relu", pooler_dropout=0.0)
    

    def forward(self, graph, type_input, func_sc_ids, return_encoder_embedding=False):
     
        bb_insts = graph.func_bb_inst_ids
        
        edge_index = graph.edge_index
        batch = graph.batch
       
        x = self.pretrained_bert(**bb_insts) # batch_size_bb_num * seq_len * hidden_dim
       
        x = x.last_hidden_state
        
       
        x = x[:,0,:] 
        x = self.gnn(x, edge_index) 
        x = scatter(x, batch, dim=0, reduce="sum") # batch_size * hidden_dim
        
        if self.config.use_string_and_const:
            func_sc_ids = self.linear1(func_sc_ids)
            x = torch.cat([x, func_sc_ids], -1) 
        
        if self.config.use_type:
            type_input = self.linear2(type_input)
            x = torch.cat([x, type_input], dim=-1)
      
        if not return_encoder_embedding and self.config.use_projector:
          
            x = self.embedding_head(x)
       
        return  x   

class EmbeddingHead(nn.Module):
    """Head for function embeddings."""
    def __init__(
            self,
            input_dim,
            output_dim,
            activation_fn,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim, bias=True)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
   
        self.out_proj = nn.Linear(input_dim, output_dim, bias=True)
        

    def forward(self, x, **kwargs):
        # print(x.shape)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = F.normalize(x)
        return x



def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
