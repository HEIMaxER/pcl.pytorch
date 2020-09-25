from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.pcl.pcl import PCL
from model.pcl_losses.functions.pcl_losses import PCLLosses
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.pcl_heads as pcl_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.vgg_weights_helper as vgg_utils
import openset.cluster_loss as cluster_loss

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses = [PCLLosses() for i in range(cfg.REFINE_TIMES)]

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, labels):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels)

    def _forward(self, data, rois, labels):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        box_feat = self.Box_Head(blob_conv, rois)
        mil_score = self.Box_MIL_Outs(box_feat)
        refine_score = self.Box_Refine_Outs(box_feat)

        if self.training:
            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            # refinement loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            for i_refine, refine in enumerate(refine_score):
                if i_refine == 0:
                    pcl_output = PCL(boxes, mil_score, im_labels, refine)
                else:
                    pcl_output = PCL(boxes, refine_score[i_refine - 1],
                                     im_labels, refine)

                refine_loss = self.Refine_Losses[i_refine](refine,
                                                           Variable(torch.from_numpy(pcl_output['labels'])),
                                                           Variable(torch.from_numpy(pcl_output['cls_loss_weights'])),
                                                           Variable(torch.from_numpy(pcl_output['gt_assignment'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_labels'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_probs'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_count'])),
                                                           Variable(torch.from_numpy(pcl_output['img_cls_loss_weights'])),
                                                           Variable(torch.from_numpy(pcl_output['im_labels_real'])))

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score

        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoICrop':
            grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                xform_out = F.max_pool2d(xform_out, 2, 2)
        elif method == 'RoIAlign':
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value


class Sim_RCNN(nn.Module):
    def __init__(self, sim_dim=5, sim_eval=0, cluster_loss_factor=1):
        super().__init__()

        self.sim_eval = sim_eval
        self.cluster_loss_factor = cluster_loss_factor

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale, sim_dim)
        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses = [PCLLosses() for i in range(cfg.REFINE_TIMES)]

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, labels):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels)

    def _forward(self, data, rois, labels):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables
        blob_conv = self.Conv_Body(im_data)
        if not self.training:
            return_dict['blob_conv'] = blob_conv

        box_feat, sim_mat, rois = self.Box_Head(blob_conv, rois)
        mil_score = self.Box_MIL_Outs(box_feat)
        refine_score = self.Box_Refine_Outs(box_feat)

        if self.training:
            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            if self.sim_eval == 0:
                clust_loss = self.cluster_loss_factor * cluster_loss.BCE_loss(mil_score, sim_mat)
                return_dict['losses']['cluster_loss_mil'] = clust_loss
            elif self.sim_eval == 1:
                clust_loss = self.cluster_loss_factor * cluster_loss.BCE_loss(refine_score[-1], sim_mat)
                return_dict['losses']['cluster_loss_refine'] = clust_loss
            elif self.sim_eval == 2:
                clust_loss_1 = self.cluster_loss_factor * cluster_loss.BCE_loss(mil_score, sim_mat)
                return_dict['losses']['cluster_loss_mil'] = clust_loss_1
                clust_loss_2 = self.cluster_loss_factor * cluster_loss.BCE_loss(refine_score[-1], sim_mat)
                return_dict['losses']['cluster_loss_refine'] = clust_loss_2


            # refinement loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            for i_refine, refine in enumerate(refine_score):
                if i_refine == 0:
                    pcl_output = PCL(boxes, mil_score, im_labels, refine)
                else:
                    pcl_output = PCL(boxes, refine_score[i_refine - 1],
                                     im_labels, refine)


                refine_loss = self.Refine_Losses[i_refine](refine,
                                                           Variable(torch.from_numpy(pcl_output['labels'])),
                                                           Variable(torch.from_numpy(pcl_output['cls_loss_weights'])),
                                                           Variable(torch.from_numpy(pcl_output['gt_assignment'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_labels'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_probs'])),
                                                           Variable(torch.from_numpy(pcl_output['pc_count'])),
                                                           Variable(torch.from_numpy(pcl_output['img_cls_loss_weights'])),
                                                           Variable(torch.from_numpy(pcl_output['im_labels_real'])))

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score

        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoICrop':
            grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                xform_out = F.max_pool2d(xform_out, 2, 2)
        elif method == 'RoIAlign':
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
