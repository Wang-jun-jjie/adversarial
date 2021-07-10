import os 
from distutils.dir_util import copy_tree

trainingset_path='../ImageNet/train'
validationset_path='../ImageNet/val'
class_mapping='../ImageNet/LOC_synset_mapping.txt'
output_path='../Restricted_ImageNet'

classes = list()
with open(class_mapping) as class_file:
    for line in class_file:
        class_dir = line[:9]
        classes.append(class_dir)
classes = tuple(classes)

if not os.path.exists(output_path):
        os.makedirs(output_path)

def id_2_foldername(definition):
    folder_list = []
    # convert the label id into folder name
    for i in definition:
        folder_list.append(classes[i])
    return folder_list

def add_class(name, definition):
    definition = id_2_foldername(definition)
    # if the class folder didn't exist, create one
    dest_dir = output_path+'/train/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i in definition:
        src_dir = trainingset_path+'/'+i
        copy_tree(src_dir, dest_dir)
    dest_dir = output_path+'/val/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i in definition:
        src_dir = validationset_path+'/'+i
        copy_tree(src_dir, dest_dir)
    
add_class('birds', list(range(10,14+1)))
add_class('turtles', list(range(33,37+1)))
add_class('lizards', list(range(42,46+1)))
add_class('spiders', list(range(72,76+1)))
add_class('crabs', list(range(118,122+1)))
add_class('dogs', list(range(205,209+1)))
add_class('cats', list(range(281,285+1)))
add_class('bigcats', list(range(289,293+1)))
add_class('beetles', list(range(302,306+1)))
add_class('butterflies', list(range(322,326+1)))
add_class('monkeys', list(range(371,374+1)))
add_class('fish', list(range(393,397+1)))
add_class('fungus', list(range(992,996+1)))
add_class('snake', list(range(60,64+1)))

add_class('musical-instrument', [402,420,486,546,594])
add_class('sportsball', [429,430,768,805,890])
add_class('cars-trucks', [609,656,717,734,817])
add_class('train', [466,547,565,820,829])
add_class('clothing', [617,834,869,841,474])
add_class('boat', [403,510,554,625,628])