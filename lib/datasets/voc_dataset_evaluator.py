# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import shutil
import uuid

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from datasets.voc_eval import voc_eval, f1_classification_score
from datasets.dis_eval import dis_eval
from utils.io import save_object
from openset.data import make_openset

logger = logging.getLogger(__name__)

def eval_classification(json_dataset,
    detected_class_ids,
    output_dir,
    use_salt=True,
    cleanup=True,
    test_corloc=False,
    use_matlab=True, seed=None, unkwn_nbr=None, mode=None, threshold=None):
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_classification_files(json_dataset, detected_class_ids, salt, seed=seed, unkwn_nbr=unkwn_nbr, mode=mode)
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    if seed != None and unkwn_nbr != None and mode != None:
        path = image_set_path.split('/')
        opensets_path = path[:-2]
        opensets_path.append("opensets")
        opensets_path = '/'.join(opensets_path)
        set_dir = '/'.join(path[:-1])
        image_set_path = make_openset(set_dir, opensets_path, unkwn_nbr, seed) + '/' + mode + '.txt'

    devkit_path = info['devkit_path']
    if seed != None and unkwn_nbr != None and mode != None and threshold != None:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}_{}_{}'.format(year, unkwn_nbr, seed, threshold))
    elif seed != None and unkwn_nbr != None and mode != None:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}_{}'.format(year, unkwn_nbr, seed))
    else:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}'.format(year))
    f1s = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(json_dataset, salt).format(cls)
        precision, recall, f1 = f1_classification_score(filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric, seed=seed, unkwn_nbr=unkwn_nbr)
        f1s += [f1]
        logger.info('f1 score for {} = {:.4f}'.format(cls, f1))
        res_file = os.path.join(output_dir, cls + '_classification_pr.pkl')
        save_object({'pre': precision, 'prec': recall, 'f1': f1}, res_file)


    logger.info('Mean f1 = {:.4f}'.format(np.mean(f1s)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for f1 in f1s:
        logger.info('{:.3f}'.format(f1))
    logger.info('{:.3f}'.format(np.mean(f1s)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')

def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=True,
    cleanup=True,
    test_corloc=False,
    use_matlab=True, seed=None, unkwn_nbr=None, mode=None, threshold=None
):
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_files(json_dataset, all_boxes, salt, seed=seed, unkwn_nbr=unkwn_nbr, mode=mode)
    if test_corloc:
        _eval_discovery(json_dataset, salt, output_dir)
    else:
        _do_python_eval(json_dataset, salt, output_dir, seed=seed, unkwn_nbr=unkwn_nbr, mode=mode, threshold=threshold)
        if use_matlab:
            print('matlab')
            _do_matlab_eval(json_dataset, salt, output_dir)
    if cleanup:
        for filename in filenames:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return None


def _write_voc_results_files(json_dataset, all_boxes, salt, seed=None, unkwn_nbr=None, mode=None):
    filenames = []
    image_set_path = voc_info(json_dataset)['image_set_path']
    if seed != None and unkwn_nbr != None and  mode != None:
        path = image_set_path.split('/')
        opensets_path = path[:-2]
        opensets_path.append("opensets")
        opensets_path = '/'.join(opensets_path)
        set_dir = '/'.join(path[:-1])
        image_set_path = make_openset(set_dir, opensets_path, unkwn_nbr, seed)+'/'+mode+'.txt'

    assert os.path.exists(image_set_path), \
        'Image set path does not exist: {}'.format(image_set_path)
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]
    # Sanity check that order of images in json dataset matches order in the
    # image set
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset,
                                                  salt).format(cls)
        filenames.append(filename)
        assert len(all_boxes[cls_ind + 1]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind + 1][im_ind]
                if type(dets) == list:
                    assert len(dets) == 0, \
                        'dets should be numpy.ndarray or empty list'
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return filenames

def _write_voc_results_classification_files(json_dataset, detected_class_ids, salt, seed=None, unkwn_nbr=None, mode=None, classification=False):
    filenames = []
    image_set_path = voc_info(json_dataset)['image_set_path']
    if seed != None and unkwn_nbr != None and  mode != None:
        path = image_set_path.split('/')
        opensets_path = path[:-2]
        opensets_path.append("opensets")
        opensets_path = '/'.join(opensets_path)
        set_dir = '/'.join(path[:-1])
        image_set_path = make_openset(set_dir, opensets_path, unkwn_nbr, seed)+'/'+mode+'.txt'

    assert os.path.exists(image_set_path), \
        'Image set path does not exist: {}'.format(image_set_path)
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]
    # Sanity check that order of images in json dataset matches order in the
    # image set
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset,
                                                  salt, classification=True).format(cls)
        filenames.append(filename)
        assert len(detected_class_ids[cls_ind + 1]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = detected_class_ids[cls_ind + 1][im_ind]
                if type(dets) == list:
                    assert len(dets) == 0, \
                        'dets should be numpy.ndarray or empty list'
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    print(dets)
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return filenames


def _get_voc_results_file_template(json_dataset, salt, classification=False):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    devkit_path = info['devkit_path']
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    if classification:
        filename = 'comp4' + salt + '_cls_' + image_set + '_{:s}.txt'
    else:
        filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    dirname = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return os.path.join(dirname, filename)


def _eval_discovery(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    cachedir = os.path.join(devkit_path, 'annotations_dis_cache_{}'.format(year))
    corlocs = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        corloc = dis_eval(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5)
        corlocs += [corloc]
        logger.info('CorLoc for {} = {:.4f}'.format(cls, corloc))
        res_file = os.path.join(output_dir, cls + '_corloc.pkl')
        save_object({'corloc': corloc}, res_file)
    logger.info('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for corloc in corlocs:
        logger.info('{:.3f}'.format(corloc))
    logger.info('{:.3f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')


def _do_python_eval(json_dataset, salt, output_dir='output', seed=None, unkwn_nbr=None, mode=None, threshold=None):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    if seed != None and unkwn_nbr != None and  mode != None:
        path = image_set_path.split('/')
        opensets_path = path[:-2]
        opensets_path.append("opensets")
        opensets_path = '/'.join(opensets_path)
        set_dir = '/'.join(path[:-1])
        image_set_path = make_openset(set_dir, opensets_path, unkwn_nbr, seed)+'/'+mode+'.txt'

    devkit_path = info['devkit_path']
    if seed != None and  unkwn_nbr != None and mode != None and threshold != None:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}_{}_{}'.format(year, unkwn_nbr, seed, threshold))
    elif seed != None and  unkwn_nbr != None and mode != None:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}_{}_{}'.format(year, unkwn_nbr, seed))
    else:
        cachedir = os.path.join(devkit_path, 'annotations_cache_{}'.format(year))
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        rec, prec, ap = voc_eval(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric, seed=seed, unkwn_nbr=unkwn_nbr)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for ap in aps:
        logger.info('{:.3f}'.format(ap))
    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')


def _do_matlab_eval(json_dataset, salt, output_dir='output'):
    import subprocess
    logger.info('-----------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code.')
    logger.info('-----------------------------------------------------')
    info = voc_info(json_dataset)
    path = os.path.join(
        cfg.ROOT_DIR, 'lib', 'datasets', 'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
       .format(info['devkit_path'], 'comp4' + salt, info['image_set'],
               output_dir)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)


def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = json_dataset.name[9:]
    devkit_path = DATASETS[json_dataset.name][DEVKIT_DIR]
    assert os.path.exists(devkit_path), \
        'Devkit directory {} not found'.format(devkit_path)
    anno_path = os.path.join(
        devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    image_set_path = os.path.join(
        devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
    return dict(
        year=year,
        image_set=image_set,
        devkit_path=devkit_path,
        anno_path=anno_path,
        image_set_path=image_set_path)
