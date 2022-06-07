from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    infoNCE loss
    """
    def __init__(self, tmp=0.05, num_sent=2, hard_negative_weight=0.0):
        super(InfoNCELoss, self).__init__()
        self.tmp = tmp 
        self.num_sent = num_sent
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, embeddings, device="cpu"):
        embeddings = embeddings.view(-1, self.num_sent, embeddings.shape[-1])
        # print(embeddings.shape)
        z1, z2 = embeddings[:,0], embeddings[:,1]
        if self.num_sent == 3:
            z3 = embeddings[:,2]
        cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.tmp
        
        z1_z1_cos = F.cosine_similarity(z1.unsqueeze(1), z1.unsqueeze(0), dim=-1) / self.tmp
        
        z1_z1_cos = z1_z1_cos-torch.eye(z1_z1_cos.shape[0],device=z1_z1_cos.device) * 1e12
        cos_sim = torch.cat([cos_sim, z1_z1_cos], 1)
        
        # Hard negative
       
        if self.num_sent >= 3:
            z1_z3_cos = F.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0), dim=-1) / self.tmp
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
        
        labels = torch.arange(cos_sim.shape[0]).long() 
        if cos_sim.cuda:
            labels = labels.to(cos_sim.device)
         # Calculate loss with hard negatives
        if self.num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights
        # print(cos_sim.shape, labels.shape)
        loss = F.cross_entropy(cos_sim, labels)  
        
        cos_sim_z2_z1 = F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=-1) / self.tmp
        z2_z2_cos = F.cosine_similarity(z2.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.tmp
        
        z2_z2_cos = z2_z2_cos-torch.eye(z2_z2_cos.shape[0],device=z2_z2_cos.device) * 1e12
        cos_sim_z2_z1 = torch.cat([cos_sim_z2_z1, z2_z2_cos], 1)
        
        loss_2 = F.cross_entropy(cos_sim_z2_z1, labels) 
        
        loss = (loss+loss_2)/2
        
        return loss

