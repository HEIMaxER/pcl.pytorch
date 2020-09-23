import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np
import math

def BCE_loss(box_cls_scores, sim_mat):
    # box_cls_scores = box_cls_scores.clamp(1e-6, 1 - 1e-6)
    for i in range(box_cls_scores.size(0)):
        for j in range(box_cls_scores.size(1)):
            if box_cls_scores[i][j] <= 0:
                print('yes yes yes yes', box_cls_scores[i][j])
    t_box_cls_scores = torch.transpose(box_cls_scores, 0, 1)
    loss = - sim_mat * torch.log(torch.mm(box_cls_scores, t_box_cls_scores)) - (1 - sim_mat) * torch.log(1 - torch.mm(box_cls_scores, t_box_cls_scores))
    loss = loss.mean()
    if math.isnan(loss):
        print("NANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANANAN")
        print('cls proba', box_cls_scores)
        print('cls proba transpose', t_box_cls_scores)
        print('sim mat', sim_mat)
    return loss