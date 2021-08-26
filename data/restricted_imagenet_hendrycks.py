import os
import argparse
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='make restricted dataset')
parser.add_argument('--ori-train', default='/tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/train', type=str, help='imagenet training dataset')
parser.add_argument('--ori-val',   default='/tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/val',   type=str, help='imagenet validation dataset')
parser.add_argument('--rst-root',  default='/tmp2/dataset/Restricted_ImageNet_Hendrycks',              type=str, help='restricted imagenet root directory')
args = parser.parse_args()

'''
class definitions
'''
Dog        = ['n02085620', 'n02099601', 'n02106550', 'n02106662', 'n02110958']
Aquatic    = ['n01910747', 'n01914609', 'n01924916', 'n02317335', 'n02655020']
Bird       = ['n01531178', 'n01558993', 'n01833805', 'n01847000', 'n01855672']
Insect     = ['n02165456', 'n02190166', 'n02219486', 'n02236044', 'n02268443']
Musical    = ['n02676566', 'n02787622', 'n02992211', 'n03840681', 'n04141076']
Vehicle    = ['n02701002', 'n03417042', 'n03445924', 'n03594945', 'n04146614']
Sport      = ['n02802426', 'n02879718', 'n04019541', 'n04039381', 'n04540053']
Building   = ['n02793495', 'n02814860', 'n02814860', 'n03788195', 'n04366367']
Food       = ['n07583066', 'n07718472', 'n07734744', 'n07753592', 'n12144580']
Appliances = ['n03196217', 'n03483316', 'n03584829', 'n04442312', 'n04554684']

def add_class(name, subclass_list):
    # training dataset
    dest_dir = args.rst_root+'/train/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for c in subclass_list:
        src_dir = args.ori_train+'/'+c
        copy_tree(src_dir, dest_dir)
    # validation dataset
    dest_dir = args.rst_root+'/val/'+name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for c in subclass_list:
        src_dir = args.ori_val+'/'+c
        copy_tree(src_dir, dest_dir)
# restricted imagenet based on Hendrycks imagenet-A w/ 200 classes
def main():
    if not os.path.exists(args.rst_root):
        os.makedirs(args.rst_root)
    add_class('Dog', Dog)
    add_class('Aquatic',Aquatic)
    add_class('Bird',Bird)
    add_class('Insect',Insect)
    add_class('Musical',Musical)
    add_class('Vehicle',Vehicle)
    add_class('Sport',Sport)
    add_class('Building',Building)
    add_class('Food',Food)
    add_class('Appliances',Appliances)
    
if __name__ == "__main__":
    main()