import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np

def BCE_loss(box_cls_scores, sim_mat):
    print("sim mat shape : ", sim_mat.shape)
    print("box scores shape : ", box_cls_scores.shape)
    t_box_cls_scores = torch.transpose(box_cls_scores, 0, 1)
    print("transpose box scores shape : ", t_box_cls_scores.shape)
    loss = -sim_mat * torch.log(t_box_cls_scores * box_cls_scores) - (1 - sim_mat) * np.log(1 - t_box_cls_scores * box_cls_scores)
    return loss.mean()