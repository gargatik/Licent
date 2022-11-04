from __future__ import print_function
import os
import argparse
from glob import glob
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import LLE
from utils import *

tf.reset_default_graph()
parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, help='1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='patch size')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./Enhanced_result', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='directory for testing inputs')
args = parser.parse_args()

def main(_):
    if args.use_gpu ==1:
        print("[*] GPU mode\n")
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = LLE(sess)
            if args.phase == 'train':
                LiCENt_train(model)
            elif args.phase == 'test':
                LiCENt_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU mode\n")
        with tf.Session() as sess:
            model = LLE(sess)
            if args.phase == 'train':
                LiCENt_train(model)
            elif args.phase == 'test':
                LiCENt_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

def LiCENt_train(LLE):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    train_low_data = []
    train_high_data = []
    train_low_data_names = glob(r'/home/atik/Desktop/Low_light/dataset/RGB/Low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob(r'/home/atik/Desktop/Low_light/dataset/RGB/Normal/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))
    print('Prepare training data')
    for idx in range(len(train_low_data_names)):
        if (idx + 1) % 100 == 0:
            print("%d / %d" % ( idx + 1, len(train_low_data_names)))
        low_im = load_img(train_low_data_names[idx],flag = 0)
        low_im = low_im[ :, :, np.newaxis]
        train_low_data.append(low_im)
        high_im = load_img(train_high_data_names[idx],flag = 0)
        high_im = high_im[ :, :, np.newaxis]
        train_high_data.append(high_im)
    print("Training Data Ready, Total traing data: %d " % (idx+1))
    LLE.train(train_low_data, train_high_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch,  ckpt_dir=args.ckpt_dir)


def LiCENt_test(LLE):    
    if args.test_dir == None:
        print("[!] NO TESTING DATA")
        exit(0)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []
    test_low_data_l = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_img(test_low_data_name[idx],flag = 1)
        test_low_im = cv2.cvtColor(test_low_im, cv2.COLOR_BGR2HLS)
        l = cv2.split(test_low_im)[1]
        l = np.array(l, dtype="float32")/255.0
        test_low_im_l = l[ :, :, np.newaxis]
        test_low_hls = test_low_im[ : , :]
        test_low_data_l.append(test_low_im_l)
        test_low_data.append(test_low_hls)	        
    LLE.test(test_low_data_l, test_low_data, test_low_data_name, save_dir=args.save_dir)

if __name__ == '__main__':
    tf.app.run()
