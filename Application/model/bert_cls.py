# coding:utf-8
from transformers import BertConfig, BertForPreTraining, BertModel, BertPreTrainedModel
from transformers.data.data_collator import default_data_collator
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from .quant_noise import quant_noise as apply_quant_noise_
import numpy as np


# for function signature preidction
class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
           
            self.config.__dict__.update(model_args.__dict__)
        self.bert = BertModel(config)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.clr_targets = self.config.clr_targets
        self.clr_nums = self.config.clr_nums
        self.reg_targets = self.config.reg_targets
        self.clr_heads = nn.ModuleDict()
        assert len(self.clr_nums) == len(self.clr_targets)
        if len(self.clr_targets) > 0:
            for i,target in enumerate(self.clr_targets):
                self.clr_heads[target+"_clr"] = Classification_head(self.config.hidden_size, self.config.inner_dim, self.clr_nums[i], activation_fn="gelu" ,pooler_dropout=0.2)
     
        self.init_weights()
    
    def forward(
            self,
            input_ids=None, # inst_num * seq_len
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, # batch_size, target_num，multi labels
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0] # cls embedding
        sequence_output = self.dropout(outputs[0])
        # print(sequence_output.shape)
        all_logits = []
        for target in self.clr_targets:
            logits = self.clr_heads[target+"_clr"](sequence_output) # batch_size, config.num_labels
            all_logits.append(logits) # list[ (batch_size, num_labels)]

        
        total_loss = 1e-6
        if labels is not None:
            assert labels.shape[1] == len(all_logits)
            
            for i in range(len(self.clr_targets)):
                loss = None
                label = labels[:,i] # batch_size, target_num
           
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(all_logits[i].view(-1, self.config.clr_nums[i]), label.view(-1))
            
                total_loss += loss

        if not return_dict:
            output = (all_logits,) + outputs[2:]
            return ((total_loss,) + output) if loss is not None else output
    
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=all_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Classification_head(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        # gelu
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
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

