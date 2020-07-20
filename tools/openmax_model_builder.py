import argparse
import cv2
import os
import pprint
import sys
import time

import _init_paths
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
from utils.io import save_object

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

os.chdir("/data1/data/expes/stage_mplocharski/pcl.pytorch")

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')
    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')


    return parser.parse_args()



def make_opemax_model(model_path, unkwn_nbr, seed):
    pass

def get_mean_activation_vectors(dataset_name, unkwn_nbr, seed, model, layer):
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES

def weibull_fit():
    pass

def main():
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)
    print('Called with args:')
    print(args)
    ds_info = args.dataset.split('.')
    ds_info[0] = ds_info[0].split("_")
    seed = ds_info[0][-1]
    unkwn_nbr = int(ds_info[0][-2])
    ds_name = ''.join(ds_info[0][0:2])

    model_name = args.load_ckpt.split('/')[-1]
    if seed not in model_name.split('_') or str(unkwn_nbr) not in model_name.split('_'):
        raise ValueError("Open dataset and model don't match.")

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if ds_name == "coco2014":
        cfg.TRAIN.DATASETS = ('coco_2014_train',)
        cfg.MODEL.NUM_CLASSES = 80 - unkwn_nbr
    elif ds_name == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 80 - unkwn_nbr
    elif ds_name == 'voc2007':
        cfg.TRAIN.DATASETS = ('voc_2007_trainval',)
        cfg.MODEL.NUM_CLASSES = 20 - unkwn_nbr
    elif ds_name == 'voc2012':
        cfg.TRAIN.DATASETS = ('voc_2012_trainval',)
        cfg.MODEL.NUM_CLASSES = 20 - unkwn_nbr
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))
    cfg.MODEL.NUM_CLASSES_OPEN = cfg.MODEL.NUM_CLASSES + 1
    print("Number of classes : ", cfg.MODEL.NUM_CLASSES)
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.NUM_GPUS = torch.cuda.device_count()

    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])
        mapping, _ = model.detectron_weight_mapping
        print(mapping)


if __name__ == '__main__':
    main()