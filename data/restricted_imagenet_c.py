import os
from pathlib import Path
import argparse
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='make restricted dataset')
parser.add_argument('--dataset', default='/tmp2/dataset/imagenet-c/noise/gaussian_noise',            type=str, help='one type of distortion in imagenet c dataset')
parser.add_argument('--result',  default='/tmp2/dataset/Restricted_ImageNet_C/noise/gaussian_noise', type=str, help='output directory of restricted imagenet dataset')
args = parser.parse_args()

'''
class definitions
'''
classes = {
    'Dog'        : ['n02085620', 'n02099601', 'n02106550', 'n02106662', 'n02110958'],
    'Aquatic'    : ['n01910747', 'n01914609', 'n01924916', 'n02317335', 'n02655020'],
    'Bird'       : ['n01531178', 'n01558993', 'n01833805', 'n01847000', 'n01855672'],
    'Insect'     : ['n02165456', 'n02190166', 'n02219486', 'n02236044', 'n02268443'],
    'Musical'    : ['n02676566', 'n02787622', 'n02992211', 'n03840681', 'n04141076'],
    'Vehicle'    : ['n02701002', 'n03417042', 'n03445924', 'n03594945', 'n04146614'],
    'Sport'      : ['n02802426', 'n02879718', 'n04019541', 'n04039381', 'n04540053'],
    'Building'   : ['n02793495', 'n02814860', 'n02814860', 'n03788195', 'n04366367'],
    'Food'       : ['n07583066', 'n07718472', 'n07734744', 'n07753592', 'n12144580'],
    'Appliances' : ['n03196217', 'n03483316', 'n03584829', 'n04442312', 'n04554684'],}

# ImageNet-C consist of testing dataset only, but there's many testing datasets
# you need a script for converting the entire imagenet-C
# the code only deals with one types of distortion, there're still 5 level of serverities.

def main():
    # get the dataset name
    dataset = Path(args.dataset)
    if dataset.exists():
        print('processing dataset name: {}'.format(dataset.name))
    else:
        print('no such dataset exist, maybe you have the wrong path')
        return
    # root of given types of distortions
    result = Path(args.result)
    result.mkdir(parents=True, exist_ok=True)
    # 5 level of serverities,
    for s in range(1, 6):
        path_in  = dataset / str(s)
        path_out = result / str(s)
        path_out.mkdir(parents=True, exist_ok=True)
        for c in classes:
            dst = path_out / c
            dst.mkdir(parents=True, exist_ok=True)
            for n in classes[c]:
                src = path_in / n
                copy_tree(str(src), str(dst))
        

if __name__ == "__main__":
    main()