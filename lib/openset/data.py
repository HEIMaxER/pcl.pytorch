import numpy as np
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



def make_openset(set_dir, opensets_path, unkwn_nbr, seed):
    """
    Generates openset labelling from VOC Imageset
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
