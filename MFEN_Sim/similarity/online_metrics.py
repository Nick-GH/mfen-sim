# encoding:utf-8
from math import cos
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import scipy.stats

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def compute_metircs(distances, labels, threshold=0.5):

    preds = distances > threshold
    labels = labels > 0

    metrics = {}
    metrics["acc"] = accuracy_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds) 
    metrics["recall"] = recall_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)

    metrics["AUC"] = roc_auc_score(labels, distances)
    metrics["threshold"] = threshold
    return metrics




class InBatchPairMetric:
    def __init__(self, num_sent=2):
        self.num_sent = num_sent
        pass

    def metrics(self, embeddings, neg_embeddings = None, return_metric=True):
        assert self.num_sent == 2 or self.num_sent == 3
        # all
        if self.num_sent == 2:
            embeddings = embeddings.view(-1, 2, embeddings.shape[-1])
            # print(embeddings.shape)
            batch_size = embeddings.shape[0]
            z1, z2 = embeddings[:,0], embeddings[:,1]
            cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) # N*N

            labels = torch.eye(batch_size).long()
            
            cos_sim = cos_sim.view(-1).detach().cpu()
            labels = labels.view(-1).cpu()
        # all
        elif self.num_sent == 3:
            if neg_embeddings == None:
                embeddings = embeddings.view(-1, 3, embeddings.shape[-1])
                z1, z2, z3 = embeddings[:,0], embeddings[:,1], embeddings[:,2]
                cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) # N,N
                z1_z3_cos = F.cosine_similarity(z1, z3, dim=-1).view(-1,1) # N,1
                cos_sim = torch.cat([cos_sim, z1_z3_cos], dim = 1) # N, N+1
                labels = torch.eye(cos_sim.shape[0], cos_sim.shape[1]).long()
                cos_sim = cos_sim.view(-1).detach().cpu()
                labels = labels.view(-1).cpu()
                
            else:
                embeddings = embeddings.view(-1, 2, embeddings.shape[-1])
                z1, z2 = embeddings[:,0], embeddings[:,1]
                cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) # N,N
                z1_z3_cos = F.cosine_similarity(z1, neg_embeddings, dim=-1).view(-1,1) # N,1
                cos_sim = torch.cat([cos_sim, z1_z3_cos], dim = 1) # N, N+1
                labels = torch.eye(cos_sim.shape[0], cos_sim.shape[1]).long()
                cos_sim = cos_sim.view(-1).detach().cpu()
                labels = labels.view(-1).cpu()
        if return_metric:

            threshold = Find_Optimal_Cutoff(np.array(labels), np.array(cos_sim))[0]
            return compute_metircs(np.array(cos_sim), np.array(labels), threshold)
        else:
            return (cos_sim, labels)

