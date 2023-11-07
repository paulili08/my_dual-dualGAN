import torch
from torch.autograd import Variable
import utils
import argparse
import os, itertools
import numpy as np
import matplotlib.pyplot as plt

"""
PARAMETERS
Batches for training : 200
Batches for testing : 200
Cascade Proportion = 0.6
batch_size = 1
The number of critic iterations per generator iteration: num_iter_G = 5
learning rate: lrD = 0.0002  lrG = 0.0002
lambda: U = 500  O = 500  V = 500
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='datasets', help='input dataset')
parser.add_argument('--datatype', required=False, default='eeg2audio', help='input dataset type')
parser.add_argument('--num_batch_train', type=int, default=200, help='num of batches training dataset divided')
parser.add_argument('--num_batch_test', type=int, default=34, help='num of batches testing dataset divided')
parser.add_argument('--batch_size', type=int, default=128, help='batch size/img size')
parser.add_argument('--cas_prop', type=float, default=0.6, help='cascade proportion')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
feat_path = r'./features'
pts = ['sub-%02d'%i for i in range(1,11)]
dire = '../Data/' + params.dataset + '/'
data_dir = params.dataset + '/'
data_dir_e2a = data_dir + params.datatype + '/'
train_dir = data_dir_e2a + 'train' + '/'
val_dir = data_dir_e2a + 'val' + '/'


if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(data_dir_e2a):
    os.mkdir(data_dir_e2a)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

# first 34 batches for testing, rest 200 for training
test = np.arange(4352)
train = np.arange(4352,29952)

for pNr, pt in enumerate(pts):
    # Load the data
    data_U = np.load(os.path.join(feat_path, f'{pt}_feat.npy'))  # eeg
    data_V = np.load(os.path.join(feat_path, f'{pt}_spec.npy'))  # audio
    pts = ['sub-%02d' % i for i in range(1, 11)]

    # Directories for saving results
    sub_train_dir = train_dir + f'{pt}' + '/'
    sub_val_dir = val_dir + f'{pt}' + '/'

    if not os.path.exists(sub_train_dir):
        os.mkdir(sub_train_dir)
    if not os.path.exists(sub_val_dir):
        os.mkdir(sub_val_dir)

    # Data pre-processing

    # Z-Normalize with mean and std from the training data
    mu = np.mean(data_U[train, :], axis=0)
    std = np.std(data_U[train, :], axis=0)
    train_data_U = (data_U[train, :] - mu) / std
    test_data_U = (data_U[test, :] - mu) / std

    mu = np.mean(data_V[train, :], axis=0)
    std = np.std(data_V[train, :], axis=0)
    train_data_V = (data_V[train, :] - mu) / std
    test_data_V = (data_V[test, :] - mu) / std
    """
    mu = np.mean(data_O[train, :], axis=0)
    std = np.std(data_O[train, :], axis=0)
    train_data_O = (data_O[train, :] - mu) / std
    test_data_O = (data_O[test, :] - mu) / std
    """
    # normalize to the range [0,1]
    train_data_U_n = (train_data_U - np.min(train_data_U)) / (np.max(train_data_U) - np.min(train_data_U))
    test_data_U_n = (test_data_U - np.min(train_data_U)) / (np.max(train_data_U) - np.min(train_data_U))
    train_data_V_n = (train_data_V - np.min(train_data_V)) / (np.max(train_data_V) - np.min(train_data_V))
    test_data_V_n = (test_data_V - np.min(train_data_V)) / (np.max(train_data_V) - np.min(train_data_V))

    # scale in range [0,255]
    train_data_U = (train_data_U_n * 255).astype(np.uint8)
    test_data_U = (test_data_U_n * 255).astype(np.uint8)
    train_data_V = (train_data_V_n * 255).astype(np.uint8)
    test_data_V = (test_data_V_n * 255).astype(np.uint8)

    # divide into several batches
    train_data_U = np.array_split(train_data_U, params.num_batch_train, axis=0)
    train_data_V = np.array_split(train_data_V, params.num_batch_train, axis=0)
    test_data_U = np.array_split(test_data_U, params.num_batch_test, axis=0)
    test_data_V = np.array_split(test_data_V, params.num_batch_test, axis=0)

    # Proportional cascade
    train_data_O = []
    for i, (U, V) in enumerate(zip(train_data_U, train_data_V)):
        eeg_part = U[0:int(params.batch_size * params.cas_prop), :]
        audio_part = V[int(params.batch_size * params.cas_prop):, :]
        train_data_O.append(np.vstack((eeg_part,audio_part)))

    test_data_O = []
    for i, (U, V) in enumerate(zip(test_data_U, test_data_V)):
        eeg_part = U[0:int(params.batch_size * params.cas_prop), :]
        audio_part = V[int(params.batch_size * params.cas_prop):, :]
        test_data_O.append(np.vstack((eeg_part, audio_part)))

    # save as .jpg
    sub_train_dir_A = sub_train_dir + 'A' + '/'
    sub_train_dir_B = sub_train_dir + 'B' + '/'
    sub_train_dir_C = sub_train_dir + 'C' + '/'
    sub_test_dir_A = sub_val_dir + 'A' + '/'
    sub_test_dir_B = sub_val_dir + 'B' + '/'
    sub_test_dir_C = sub_val_dir + 'C' + '/'

    if not os.path.exists(sub_train_dir_A):
        os.mkdir(sub_train_dir_A)
    if not os.path.exists(sub_train_dir_B):
        os.mkdir(sub_train_dir_B)
    if not os.path.exists(sub_train_dir_C):
        os.mkdir(sub_train_dir_C)
    if not os.path.exists(sub_test_dir_A):
        os.mkdir(sub_test_dir_A)
    if not os.path.exists(sub_test_dir_B):
        os.mkdir(sub_test_dir_B)
    if not os.path.exists(sub_test_dir_C):
        os.mkdir(sub_test_dir_C)

    for num, img in enumerate(train_data_U, start=1):
        filename = os.path.join(sub_train_dir_A, f'A_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')

    for num, img in enumerate(train_data_O, start=1):
        filename = os.path.join(sub_train_dir_B, f'B_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')

    for num, img in enumerate(train_data_V, start=1):
        filename = os.path.join(sub_train_dir_C, f'C_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')

    for num, img in enumerate(test_data_U, start=1):
        filename = os.path.join(sub_test_dir_A, f'A_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')

    for num, img in enumerate(test_data_O, start=1):
        filename = os.path.join(sub_test_dir_B, f'B_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')

    for num, img in enumerate(test_data_V, start=1):
        filename = os.path.join(sub_test_dir_C, f'C_{num}.jpg')
        plt.imsave(filename, img, cmap='gray')


