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
    t_box_cls_scores = torch.transpose(box_cls_scores, 0, 1)
    loss = - sim_mat * torch.log(torch.mm(box_cls_scores, t_box_cls_scores)) - (1 - sim_mat) * torch.log(1 - torch.mm(box_cls_scores, t_box_cls_scores))
    return loss.mean()