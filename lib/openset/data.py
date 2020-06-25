import numpy as np
from itertools import combinations
import scipy.special
import os
# from lib.datasets.dataset_catalog import DATASETS

set_path = "../data/VOCdevkit/VOC2007/ImageSets/Main/"
opensetpath = "../data/VOCdevkit/VOC2007/Openset/"

def get_class_labels(filename):
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
    filenames = os.listdir(set_dir)
    class_names = []
    for name in filenames:
        cls_name = name.split('_')[0]
        if '.' not in cls_name and cls_name not in class_names:
            class_names.append(cls_name)
    return class_names

def make_class_labels(label_dict, openset_dir):
    os.makedirs(openset_dir)
    class_names = list(label_dict.keys())

    for set in list(label_dict[class_names[0]].keys()):
        file_name = set+".txt"
        file_path = os.path.join(openset_dir, file_name)
        file = open(file_path, "w")
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
    perm = list(combinations(class_names, unknw_nbr))
    print('permutaion length :', len(perm))
    return list(perm[int(seed)])



def make_openset(set_dir, opensets_path, unkwn_nbr, seed):

    class_names = get_class_names(set_dir)
    print('max seed : ',scipy.special.comb(len(class_names), unkwn_nbr))
    seed %= scipy.special.comb(len(class_names), unkwn_nbr)  # la grain de l'open set correspond à un choix de classes inconnues parmis les classes labelisees
    folder_name = str(unkwn_nbr)+'_'+str(int(seed))
    openset_dir = os.path.join(opensets_path, folder_name)
    openset_dir = os.path.join(openset_dir, 'Main')
    print(openset_dir)
    if os.path.exists(openset_dir):
        print("Open set already exists")
    else:

        sets = ['train', 'trainval', 'val', 'test']
        set_separation = {}
        for set in sets:
            set_separation[set] = get_class_labels(os.path.join(set_dir, set+'.txt'))
            print(set, len(set_separation[set].keys()))

        unkwn_classes = get_unknown_classes(class_names, seed, unkwn_nbr)

        print('unknown classes : ', unkwn_classes)

        labels = {}
        for name in class_names:
            labels[name] = {}
            for set in sets:
                labels[name][set] = get_class_labels(os.path.join(set_dir, name+'_'+set+'.txt'))

        unknown = {}

        for id in list(set_separation[sets[1]].keys()):
            for cls in unkwn_classes :
                if labels[cls][sets[1]][id] == 0 or labels[cls][sets[1]][id] == 1:
                    if id in set_separation[sets[1]]:
                        set_separation[sets[1]].pop(id)
                        if id in set_separation[sets[0]]:
                            set_separation[sets[0]].pop(id)
                        else:
                            set_separation[sets[2]].pop(id)
                        set_separation[sets[3]][id] = None
                        unknown[id] = labels[cls][sets[1]][id]
        unknown_test = {}
        for id in list(set_separation[sets[-1]].keys()):
            for cls in unkwn_classes :
                try:
                    if labels[cls][sets[-1]][id] == 0 or labels[cls][sets[-1]][id] == 1:
                        unknown_test[id] = labels[cls][sets[-1]][id]
                except:
                    pass

        new_labels = {'unknown':{s:{} for s in sets}}

        for cls in class_names:
            if cls not in unkwn_classes:
                new_labels[cls] = {}
                for set in sets:
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
                            try:
                                if id not in new_labels['unknown'][set]:
                                    new_labels['unknown'][set][id] = labels[cls][old_set][id]
                                elif labels[cls][old_set][id] > new_labels['unknown'][set][id]:
                                    new_labels['unknown'][set][id] = labels[cls][old_set][id]
                            except:
                                pass

        print(len(class_names), '-', len(unkwn_classes), " + 1 =", len(new_labels.keys()))

        for set in sets:
            print(set, len(set_separation[set].keys()))


        print('unknown', len(unknown.keys()), len(unknown_test.keys()), len(new_labels['unknown']['test'].keys()))

        make_class_labels(new_labels, openset_dir)
        print("Made new Openset : ", openset_dir)



make_openset(set_path, opensetpath, 1, 325)





# print(get_class_names(set_path+"Main"))


# print(len(get_class_labels(set_path+'train.txt')), get_class_labels(set_path+'train.txt'))
# print(len(get_class_labels(set_path+'test.txt')), get_class_labels(set_path+'test.txt'))
# print(len(get_class_labels(set_path+'trainval.txt')), get_class_labels(set_path+'trainval.txt'))
# print(len(get_class_labels(set_path+'val.txt')), get_class_labels(set_path+'val.txt'))

# print(len(get_class_labels(set_path+'Main/train.txt')), len(get_class_labels(set_path+'Layout/train.txt')), len(get_class_labels(set_path+'Segmentation/train.txt')))
# print(len(get_class_labels(set_path+'Main/trainval.txt')), len(get_class_labels(set_path+'Layout/trainval.txt')), len(get_class_labels(set_path+'Segmentation/trainval.txt')))
# print(len(get_class_labels(set_path+'Main/val.txt')), len(get_class_labels(set_path+'Layout/val.txt')), len(get_class_labels(set_path+'Segmentation/val.txt')))
# print(len(get_class_labels(set_path+'Main/test.txt')), len(get_class_labels(set_path+'Layout/test.txt')), len(get_class_labels(set_path+'Segmentation/test.txt')))
#
# unique_labels = []
# for x in os.listdir(set_path):
#     labels = get_class_labels(set_path+x)
#     for i in labels.items():
#         if i[1] not in unique_labels:
#             unique_labels.append(i[1])
# print(unique_labels)


#
# labels = get_class_labels(set_path+"/aeroplane_test.txt")
#
# print(labels)
