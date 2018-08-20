
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os, sys, path, shutil

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

EPSILON = 1e-8

def init_caffe_network():
    #remove the following two lines if testing with cpu
    caffe.set_mode_gpu()
    # choose which GPU you want to use
    caffe.set_device(0)
    caffe.SGDSolver.display = 0
    # load net
    net = caffe.Net('vgg16/deploy.prototxt', 'vgg16/dss_model_released.caffemodel', caffe.TEST)
    return net

# Visualization
def plot_single_scale(scale_lst, name_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1,5,i+1)
        s.set_xlabel(name_lst[i], fontsize=10)
        if name_lst[i] == 'Source':
            plt.imshow(scale_lst[i])
        else:
            plt.imshow(scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()

def detect_saliency_for_one_image(net, img_file, result_dir, max_side=0, disp_result=False):
    img = Image.open(img_file)
    width, height = img.size
    print(width, height)
    if max_side > 0 and max(width, height) > max_side:
        scale = max_side * 1.0 / max(width,height)
        width, height = int(width*scale+0.5), int(height*scale+0.5)
        img = img.resize((width, height))
    print(width, height)
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im

    # run net and take argmax for prediction
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    res = (out3 + out4 + out5 + fuse) / 4
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)

    res = np.array(res*255, dtype=np.uint8)
    image = Image.fromarray(res)
    image_name = os.path.basename(img_file)[:-4]
    res_file = result_dir + image_name + '_res.png'
    image.save(res_file)

    img_file_dst = result_dir + image_name + '.jpg'
    shutil.copyfile(img_file, img_file_dst)

    gt_file = img_file[:-4] + '.png'
    if os.path.isfile(gt_file):
        gt_file_dst = result_dir + image_name + '.png'
        shutil.copyfile(gt_file, gt_file_dst)

    if disp_result:
        if os.path.isfile(gt_file):
            gt = Image.open(gt_file)
            gt = np.array(gt, dtype=np.uint8)
        else:
            gt = img
        out_lst = [out1, out2, out3, out4, out5]
        name_lst = ['SO1', 'SO2', 'SO3', 'SO4', 'SO5']
        plot_single_scale(out_lst, name_lst, 10)
        out_lst = [out6, fuse, res, img, gt]
        name_lst = ['SO6', 'Fuse', 'Result', 'Source', 'GT']
        plot_single_scale(out_lst, name_lst, 10)

    return res

def list_image_files(image_dir):
    image_files = []
    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)
        path = path.replace('\\', '/')
        if os.path.isfile(path):
            ext = filename[-4:]
            if ext == '.jpg':
                image_files.append(filename)
    return image_files

def detect_saliency_for_images(net, image_dir, max_side=0, disp_result=False):
    if len(image_dir) == 0:
        image_dir = '/export/zengjp/datasets/saliency/MSRA-B/images/'
        with open('data/msra_b/test.lst') as f:
            image_list = f.readlines()  
    else:
        if image_dir[-1] != '/':
            image_dir = image_dir + '/'
        image_list = list_image_files(image_dir)
    image_list = [image_dir+x.strip() for x in image_list]

    result_dir = image_dir + 'results/'
    os.makedirs(result_dir, exist_ok=True)

    for img_file in image_list:
        print('processing image: %s' % (img_file))
        detect_saliency_for_one_image(net, img_file, result_dir, max_side, disp_result)


def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Parameters for salient object detection')
    parser.add_argument(
        '--image_dir', dest='image_dir',
        help='Directory for input images(jpeg).',
        default='', type=str)
    parser.add_argument(
        '--disp_result', dest='disp_result',
        help='Whether to display result images.',
        default=False, type=bool)
    parser.add_argument(
        '--max_side', dest='max_side',
        help='Max length of image long side.',
        default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    net = init_caffe_network()
    detect_saliency_for_images(net, args.image_dir, args.max_side, args.disp_result)

