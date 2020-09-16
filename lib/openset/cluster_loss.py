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
    t_box_cls_scores = torch.transpose(box_cls_scores, 0, 1)
    print("transpose box scores shape : ", t_box_cls_scores.shape)
    print("box scores shape : ", box_cls_scores.shape)
    loss_one = torch.log(torch.mm(box_cls_scores, t_box_cls_scores))
    print("loss_one shape : ", loss_one.shape)
    print('sim_mat device :', sim_mat.device)

    sim_mat.to('cuda')
    print('sim_mat device :', sim_mat.device)

    loss_one = - sim_mat * loss_one
    loss_mone = - (1 - sim_mat) * np.log(1 - t_box_cls_scores * box_cls_scores)
    loss = loss_one + loss_mone
    return loss.mean()