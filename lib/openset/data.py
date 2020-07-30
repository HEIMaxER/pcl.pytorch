import numpy as np
import json
import pickle
from itertools import combinations
import scipy.special
import os
# from lib.datasets.dataset_catalog import DATASETS

def get_class_labels(filename):
    """
    open VOC Imageset text files with image ids and labels and puts them in a dict project
    :param filename: name of text file to extract labels from
    :return dict object
    """
    id_file = open(filename, 'r')
    lines = id_file.readlines()
    labels = {}
    for l in lines:
        vals = l[:-1].split(' ')
        if vals[0] != vals[-1] and vals[-1] != '':
            labels[vals[0]] = int(vals[-1])
        else:
            labels[vals[0]] = None
    return labels

def get_class_names(set_dir):
    """
    gets class names from a VOC Imageset Main directory
    :param set_dir: path to directory containing label files
    :return list object :
    """
    filenames = os.listdir(set_dir)
    class_names = []
    for name in filenames:
        cls_name = name.split('_')[0]
        if '.' not in cls_name and cls_name not in class_names:
            class_names.append(cls_name)
    return class_names

def make_class_labels(label_dict, openset_dir):
    """
    generates labels files from dict object
    :param label_dict: dict structured like VOC label set files
    :param openset_dir: path to the openset directory where to generate the label files
    :return:
    """
    os.makedirs(openset_dir)
    class_names = list(label_dict.keys())

    for set in list(label_dict[class_names[0]].keys()):
        file_name = set+".txt"
        file_path = os.path.join(openset_dir, file_name)
        file = open(file_path, "w")                 #generates
        keys = list(label_dict[class_names[0]][set].keys())
        keys = np.sort(keys)
        for key in keys:
            file.write(str(key)+"\n")
        file.close()

    for name in class_names:
        for set in list(label_dict[class_names[0]].keys()):
            file_name = name + "_" + set + ".txt"
            file_path = os.path.join(openset_dir, file_name)
            file = open(file_path, "w")
            keys = list(label_dict[name][set].keys())
            keys = np.sort(keys)
            for key in keys:
                if label_dict[name][set][key] != -1:
                    file.write(str(key)+"  "+ str(label_dict[name][set][key]) + "\n")
                else:
                    file.write(str(key) + " " + str(label_dict[name][set][key]) + "\n")
            file.close()



def get_unknown_classes(class_names, seed, unknw_nbr):
    """
    chooses class names to be branded as unknow based on class names and a seed corresponding to one possible pick
    :param class_names: classes to choose from
    :param seed: id of the possible cobination of unknown classes
    :param unknw_nbr: number of unknown clisses to be picked
    :return:
    """
    comb = list(combinations(class_names, unknw_nbr))
    return list(comb[seed])

def make_annotations(annotaion_fn, seed, unkwn_nbr):
    """
    Make an openset split of COCO annotation files based on the number of unknown classes and seed number
    :param annotaion_fn: Annotations file name
    :param seed: Integer used for picking unknown classes from all the classes in the dataset
    :param unkwn_nbr: number of unknown classes
    :return: new annotation file names and trainval/test split
    """
    split_path = annotaion_fn.split('/')
    split_path[-1] = split_path[-1].split('_')[:-1]

    trainval_path = split_path.copy()
    trainval_path[-1].append('trainval.json')
    trainval_path[-1] = '_'.join(trainval_path[-1])
    trainval_path = '/'.join(trainval_path)

    split_path[-1] = split_path[-1][:-1]

    test_path = split_path.copy()
    test_path[-1].append('test.json')
    test_path[-1] = '_'.join(test_path[-1])
    test_path = '/'.join(test_path)

    try:
        with open(trainval_path) as json_file:
            trainval_annotations = json.load(json_file)

        with open(test_path) as json_file:
            test_annotations = json.load(json_file)

    except:
        print("Annotation file dosen't exist")

    class_names = []

    for data in trainval_annotations['categories']:
        class_names.append(data['name'])

    seed %= scipy.special.comb(len(class_names),unkwn_nbr)  # normalizing seed as diffent unknown numbers can have a different amount of possible seeds
    seed = int(seed)

    new_trainval_path = trainval_path.split('.')[:-1]
    new_trainval_path[-1] += '_' + str(unkwn_nbr) + '_' + str(seed)
    new_trainval_path.append('json')                                      #setting new annotation file names
    new_trainval_path = '.'.join(new_trainval_path)

    new_test_path = test_path.split('.')[:-1]
    new_test_path[-1] += '_' + str(unkwn_nbr) + '_' + str(seed)
    new_test_path.append('json')
    new_test_path = '.'.join(new_test_path)
    file_names = {'trainval': new_trainval_path, 'test': new_test_path}


    if os.path.exists(new_test_path) and os.path.exists(new_trainval_path):
        print("Openset already exists")
        print(new_trainval_path)
        print(new_test_path)
        with open(new_trainval_path) as json_file:
            os_trainval_annotations = json.load(json_file)

        with open(new_test_path) as json_file:
            os_test_annotations = json.load(json_file)

        image_ids = {'trainval': [], 'test': []}
        for k in range(len(os_trainval_annotations['images'])):
            annot = os_trainval_annotations['images'][k]           #getting image ids for later proposal split
            image_ids['trainval'].append(annot['id'])
        for k in range(len(os_test_annotations['images'])):
            annot = os_test_annotations['images'][k]
            image_ids['test'].append(annot['id'])


    else:
        print("Making Openset :")
        unknw_class = get_unknown_classes(class_names, seed, unkwn_nbr) #getting unknown class names

        if unkwn_nbr > 1:
            print("{} unknown classes : ".format(unkwn_nbr), *unknw_class, sep = ", ")

        else :
            print("{} unknown classe : ".format(unkwn_nbr), unknw_class)
        unknw_ids = []

        k=0
        i=0
        for data in trainval_annotations['categories']:
            if data['name'] in unknw_class:
                unknw_ids.append(data['id'])             #Making new class ids
                class_names[k] = [class_names[k], None]
                i+=1
            else:
                class_names[k] = [class_names[k], data['id']-i]
            k+=1

        new_trainval_annotations = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}
        new_test_annotations = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}        #creating new annotation dict

        for k in range(len(class_names)):
            if class_names[k][1] != None:
                new_trainval_annotations['categories'].append({'supercategory': 'none', 'id': class_names[k][1], 'name': class_names[k][0]})
                new_test_annotations['categories'].append({'supercategory': 'none', 'id': class_names[k][1], 'name': class_names[k][0]})      #filling categories
        new_test_annotations['categories'].append(
            {'supercategory': 'none', 'id': len(new_test_annotations['categories'])+1, 'name': 'unknown'})

        image_ids = {'trainval':[], 'test':[]}

        for k in range(len(trainval_annotations['annotations'])):
            annot = trainval_annotations['annotations'][k]
            if annot['category_id'] in unknw_ids and annot['image_id'] not in image_ids['test']:
                image_ids['test'].append(annot['image_id'])
            elif annot['image_id'] not in image_ids['trainval'] and annot['image_id'] not in image_ids['test']:
                image_ids['trainval'].append(annot['image_id'])                   #spliting images based on annotations

        for k in range(len(test_annotations['annotations'])):
            annot = test_annotations['annotations'][k]
            if annot['image_id'] not in image_ids['test']:
                image_ids['test'].append(annot['image_id'])

        image_ids['test'] = list(np.unique(image_ids['test']))
        image_ids['trainval'] = list(np.unique(image_ids['trainval']))

        for id in image_ids['test']:
            if id in image_ids['trainval']:                 #removing images from training set that belong in the test set as image may contain multiple objects
                image_ids['trainval'].remove(id)

        for k in range(len(test_annotations['annotations'])):
            annot = test_annotations['annotations'][k]
            if annot['category_id'] in unknw_ids:                                   #copying and sorting annotations in test set
                annot['category_id'] = new_test_annotations['categories'][-1]['id']
            else:
                annot['category_id'] = class_names[annot['category_id']-1][1]
            new_test_annotations['annotations'].append(annot)

        for k in range(len(trainval_annotations['annotations'])):
            annot = trainval_annotations['annotations'][k]
            if annot['category_id'] in unknw_ids :
                annot['category_id'] = new_test_annotations['categories'][-1]['id']  #copying and sorting annotations in trainval set
                new_test_annotations['annotations'].append(annot)
            elif annot['image_id'] in image_ids['test']:
                annot['category_id'] = class_names[annot['category_id'] - 1][1]
                new_test_annotations['annotations'].append(annot)
            else:
                annot['category_id'] = class_names[annot['category_id']-1][1]
                new_trainval_annotations['annotations'].append(annot)

        for k in range(len(test_annotations['images'])):
            img = test_annotations['images'][k]
            if img['id'] in image_ids['test']:
                new_test_annotations['images'].append(img)     #copying and sorting image references in test set
            else :
                print('Unknown test annotaion id')

        for k in range(len(trainval_annotations['images'])):
            img = trainval_annotations['images'][k]
            if img['id'] in image_ids['test']:
                new_test_annotations['images'].append(img)      #copying and sorting image references in trainval set
            elif img['id'] in image_ids['trainval']:
                new_trainval_annotations['images'].append(img)
            else:
                print("Unknown trainval annotion id")

        with open(new_test_path, 'w') as outfile:
            json.dump(new_test_annotations, outfile)
            #writting annotations to json
        with open(new_trainval_path, 'w') as outfile:
            json.dump(new_trainval_annotations, outfile)
    return file_names, image_ids

def split_proposals(proposal_file, ids, seed, unkwn_nbr):
    """
    Splitting precalculated proposals based on an openset split
    :param proposal_file: proposal file name
    :param ids: ids split betweed trainval and test sets
    :param seed: openset seed
    :param unkwn_nbr: number of unknown classes
    :return: new proposal file name
    """
    if seed not in proposal_file and unkwn_nbr not in proposal_file:
        proposal_path = proposal_file.split('/')
        proposal_path[-1] = proposal_path[-1].split('_')[:-1]

        trainval_path = proposal_path.copy()
        trainval_path[-1].append('trainval.pkl')
        trainval_path[-1] = '_'.join(trainval_path[-1])
        trainval_path = '/'.join(trainval_path)

        proposal_path[-1] = proposal_path[-1][:-1]                      #getting basic trainval and test proposal files

        test_path = proposal_path.copy()
        test_path[-1].append('test.pkl')
        test_path[-1] = '_'.join(test_path[-1])
        test_path = '/'.join(test_path)

    else:
        proposal_path = proposal_file.split('/')
        proposal_path[-1] = proposal_path[-1].split('_')

        trainval_path = proposal_path.copy()
        trainval_path[-1][2] = 'trainval'
        trainval_path[-1] = '_'.join(trainval_path[-1])
        trainval_path = '/'.join(trainval_path)

        test_path = proposal_path.copy()
        test_path[-1][2] = 'trainval'
        test_path[-1] = '_'.join(test_path[-1])
        test_path = '/'.join(test_path)
        try:
            with open(trainval_path, 'rb') as f:
                trainval_proposals = pickle.load(f)
                # loading basic proposal files
            with open(test_path, 'rb') as f:
                test_proposals = pickle.load(f)
            print("Proposals already split")

            file_names = {'trainval': trainval_path, 'test': test_path}
            return file_names
        except:
            print("Refenced dataset doesn't have split proposal files") #TODO try to get original proposal file name to resplit it

    try:
        with open(trainval_path, 'rb') as f:
            trainval_proposals = pickle.load(f)
                                                                    #loading basic proposal files
        with open(test_path, 'rb') as f:
            test_proposals = pickle.load(f)

    except:
        print("Proposal file dosen't exist")

    new_trainval_path = trainval_path.split('.')[:-1]
    new_trainval_path[-1] += '_' + str(unkwn_nbr) + '_' + str(seed)
    new_trainval_path.append('pkl')
    new_trainval_path = '.'.join(new_trainval_path)
                                                                    #building path for new proposal files
    new_test_path = test_path.split('.')[:-1]
    new_test_path[-1] += '_' + str(unkwn_nbr) + '_' + str(seed)
    new_test_path.append('pkl')
    new_test_path = '.'.join(new_test_path)
    file_names = {'trainval': new_trainval_path, 'test': new_test_path}


    if os.path.exists(new_test_path) and os.path.exists(new_trainval_path):
        print("Proposals already sorted")
        print(new_trainval_path)
        print(new_test_path)
    else:                                                          #checking openset proposal file already exist
        print("Splitting selective search proposals proposals")
        new_test_proposals = {}
        new_trainval_proposals = {}

        for k in trainval_proposals.keys():
            new_test_proposals[k] = []
            new_trainval_proposals[k] = []

        for k in range(len(trainval_proposals['indexes'])):
            if trainval_proposals['indexes'][k] in ids['trainval']:
                for key in new_trainval_proposals.keys():
                    new_trainval_proposals[key].append(trainval_proposals[key][k])
            elif trainval_proposals['indexes'][k] in ids['test']:
                for key in new_test_proposals.keys():
                    new_test_proposals[key].append(trainval_proposals[key][k])
            else:
                print("Unknown proposal index : {}".format(trainval_proposals['indexes'][k]))
                                                                                        #splitting proposals based on ids
        for k in range(len(test_proposals['indexes'])):
            if test_proposals['indexes'][k] in ids['test']:
                for key in new_test_proposals.keys():
                    new_test_proposals[key].append(test_proposals[key][k])
            else:
                print("Test proposal not found in openset split{}".format(test_proposals['indexes'][k]))

        with open(new_test_path, 'wb') as outfile:
            pickle.dump(new_test_proposals, outfile)
            #writting proposals to pickle
        with open(new_trainval_path, 'wb') as outfile:
            pickle.dump(new_trainval_proposals, outfile)
    return file_names

def make_openset(set_dir, opensets_path, unkwn_nbr, seed):
    """
    Generates openset labelling from VOC Imageset as well as COCO annotations for the said set
    :param set_dir: path to original VOC set to generate from
    :param opensets_path: path to generate openset labels in
    :param unkwn_nbr: number of classes to make as unknown
    :param seed: id of the combination of classes chosen as unknown
    :return: openset path
    """

    class_names = get_class_names(set_dir)

    seed %= scipy.special.comb(len(class_names), unkwn_nbr) #normalizing seed as diffent unknown numbers can have a different amount of possible seeds
    seed = int(seed)

    folder_name = str(unkwn_nbr)+'_'+str(seed)
    openset_dir = os.path.join(opensets_path, folder_name) #generating openset foldername
    openset_dir = os.path.join(openset_dir, 'Main')

    if os.path.exists(openset_dir):
        print("Open set already exists")   #checking if openset already exists

    else:

        sets = ['train', 'trainval', 'val', 'test']
        set_separation = {}
        for set in sets:
            set_separation[set] = get_class_labels(os.path.join(set_dir, set+'.txt'))  #gettting general set layout

        unkwn_classes = get_unknown_classes(class_names, seed, unkwn_nbr)        #getting unknown class names


        labels = {}
        for name in class_names:
            labels[name] = {}                  #getting all original labels
            for set in sets:
                labels[name][set] = get_class_labels(os.path.join(set_dir, name+'_'+set+'.txt'))


        for id in list(set_separation[sets[1]].keys()):         # all images containing unknown class entries are moved to the testing set
            for cls in unkwn_classes :
                if labels[cls][sets[1]][id] == 0 or labels[cls][sets[1]][id] == 1:
                    if id in set_separation[sets[1]]:
                        set_separation[sets[1]].pop(id)
                        if id in set_separation[sets[0]]:       # sets[1] is the joined training and validation sets (sets[0], sets[2])
                            set_separation[sets[0]].pop(id)
                        else:
                            set_separation[sets[2]].pop(id)    # all images containing unknown classes from the trainval sets are moved to the testing set
                        set_separation[sets[3]][id] = None


        new_labels = {'unknown':{s:{} for s in sets}}

        for cls in class_names:
            if cls not in unkwn_classes:
                new_labels[cls] = {}
                for set in sets:                                 # new label files are reconstructed based on the new set separations
                    new_labels[cls][set] = {}
                    for id in set_separation[set].keys():
                        for old_set in sets:
                            try:
                                new_labels[cls][set][id] = labels[cls][old_set][id]
                            except:
                                pass
            else:
                for set in sets:
                    for id in set_separation[set].keys():
                        for old_set in sets:
                            try:                                                            # all labels from classes chosen as unknown are mashed into the new unknown class wich only has entries in the testing set
                                if id not in new_labels['unknown'][set]:
                                    new_labels['unknown'][set][id] = labels[cls][old_set][id]
                                elif labels[cls][old_set][id] > new_labels['unknown'][set][id]:
                                    new_labels['unknown'][set][id] = labels[cls][old_set][id]
                            except:
                                pass

        make_class_labels(new_labels, openset_dir)                           #generating new label files
        print("Made new Openset : ", openset_dir)
    return openset_dir
