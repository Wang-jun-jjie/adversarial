import os 
import argparse
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='make restricted dataset')
parser.add_argument('--ori-train', default='/tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/train', type=str, help='imagenet training dataset')
parser.add_argument('--ori-val',   default='/tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/val',   type=str, help='imagenet validation dataset')
parser.add_argument('--class-map', default='/tmp2/dataset/imagenet/LOC_synset_mapping.txt',    type=str, help='imagenet class mapping')
parser.add_argument('--rst-root',  default='/tmp2/dataset/Restricted_ImageNet',                type=str, help='restricted imagenet root directory')
parser.add_argument('--n-class', '-n', default=10, type=int, help='number of classes for restricted imagenet. (10, 14, or 20, default: 10)')
args = parser.parse_args()

# trainingset_path='../ImageNet/train'
# validationset_path='../ImageNet/val'
# class_mapping='../ImageNet/LOC_synset_mapping.txt'
# output_path='../Restricted_ImageNet'

def id_2_foldername(definition):
    classes = list()
    with open(args.class_map) as class_file:
        for line in class_file:
            class_dir = line[:9]
            classes.append(class_dir)
    classes = tuple(classes)
    folder_list = []
    # convert the label id into folder name
    for i in definition:
        folder_list.append(classes[i])
    return folder_list

def add_class(name, definition):
    definition = id_2_foldername(definition)
    # if the class folder didn't exist, create one
    dest_dir = args.rst_root+'/train/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i in definition:
        src_dir = args.ori_train+'/'+i
        copy_tree(src_dir, dest_dir)
    dest_dir = args.rst_root+'/val/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i in definition:
        src_dir = args.ori_val+'/'+i
        copy_tree(src_dir, dest_dir)

def main():
    if not os.path.exists(args.rst_root):
        os.makedirs(args.rst_root)
    if args.n_class != 10 and args.n_class != 14 and args.n_class != 20:
        print('config not supported')
        return

    if args.n_class >= 10:
        add_class('birds', list(range(10,14+1)))
        add_class('turtles', list(range(33,37+1)))
        add_class('lizards', list(range(42,46+1)))
        add_class('spiders', list(range(72,76+1)))
        add_class('crabs', list(range(118,122+1)))
        add_class('dogs', list(range(205,209+1)))
        add_class('beetles', list(range(302,306+1)))
        add_class('butterflies', list(range(322,326+1)))
        add_class('fish', list(range(393,397+1)))
        add_class('fungus', list(range(992,996+1)))
    if args.n_class >= 14:
        add_class('monkeys', list(range(371,374+1)))
        add_class('snake', list(range(60,64+1)))
        add_class('cats', list(range(281,285+1)))
        add_class('bigcats', list(range(289,293+1)))
    if args.n_class >= 20:
        add_class('musical-instrument', [402,420,486,546,594])
        add_class('sportsball', [429,430,768,805,890])
        add_class('cars-trucks', [609,656,717,734,817])
        add_class('train', [466,547,565,820,829])
        add_class('clothing', [617,834,869,841,474])
        add_class('boat', [403,510,554,625,628])

if __name__ == "__main__":
    main()