import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils

def BCE_loss(box_cls_scores, sim_mat):
    M =  box_cls_scores.size(0)
    loss = torch.zeros(1)
    print("sim mat shape : ", sim_mat.shape)
    print("box scores shape : ", box_cls_scores.shape)
    for i in range(M):
        for j in range(M):
            cls_score_i = box_cls_scores[i]
            cls_score_j = box_cls_scores[j]

            cls_score_i = cls_score_i.clamp(1e-6, 1 - 1e-6)
            cls_score_j = cls_score_j.clamp(1e-6, 1 - 1e-6)

            loss -= sim_mat[i][j] * torch.log(torch.transpose(cls_score_i, 0, 1) * cls_score_j) + (1 - sim_mat[i][j]) * torch.log(1 - torch.transpose(cls_score_i, 0, 1) * cls_score_j)

    loss *= (1/M**2)

    return loss.mean()