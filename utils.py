import numpy as np
import tensorflow as tf
import random
from scipy import ndimage
import os
import re
import pandas as pd
import nibabel as nib
from read_img import read_nifti, write_nifti
from PIL import Image

def get_random_2D_slice(X, rand_start=0, rand_end=200):
    np.random.choice(rand_start, rand_end)

def gaussian_normalize(X):
    mean = np.mean(X)
    std = np.std(X)

    X -= mean
    X /= std
    return X


def max_normalize(X):
    X /= np.max(X)  # 16bit data
    return X


def minmax_normalize(X, min_val=0, max_val=2**16, iba=False):
    if iba:
        max_val = 2**8
    X = (X - min_val) / (max_val - min_val)
    return X


def create_overlay(pred, target):
    a = target.copy()
    b = pred.copy()
    a[a > 1] = 1
    b[b > 1] = 1
    res = a*2 + b
    res[res==2] = 10
    res[res==3] = 2
    res[res==10]= 3
    return res



def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-270, -180, -90, 90, 180, 270]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_3D_slices(X, old_len, new_len):
    if old_len <= new_len:
        raise ValueError("Check your lengths!")
    if old_len % new_len != 0:
        raise ValueError("New length should be a factor of old len")

    output = []
    for k in range(0, old_len, new_len):
        for j in range(0, old_len, new_len):
            for i in range(0, old_len, new_len):
                output.append(X[i:i+new_len, j:j+new_len, k:k+new_len])

    return output




def get_spliced_arrays(X, y, splice_by=4, normalize='minmax'):
    X_spliced, y_spliced = [], []
    for i in range(len(X)):
        X_spliced.extend(np.array(get_3D_slices(X[i], int(X.shape[-1]), int(X.shape[-1] / splice_by))))
        y_spliced.extend(np.array(get_3D_slices(y[i], int(y.shape[-1]), int(y.shape[-1] / splice_by))))
    if normalize == 'minmax':
        X_spliced = [minmax_normalize(arr) for arr in X_spliced]
    X_spliced = np.array(X_spliced)
    y_spliced = np.array(y_spliced)
    return X_spliced, y_spliced

def reshape_spliced_arrays(X, old_len=200, new_len=100, pred=True):
    output_array = []
    temp_array = np.reshape(X, (-1, old_len, old_len, old_len))
    x = 0
    while x < X.shape[0]:
        if pred:
            output = np.zeros((old_len, old_len, old_len, 1))
        else:
            output = np.zeros((old_len, old_len, old_len))
        y = 0

        while y < X.shape[0]/temp_array.shape[0]:
            for k in range(0, old_len, new_len):
                for j in range(0, old_len, new_len):
                    for i in range(0, old_len, new_len):
                        output[i:i + new_len, j:j + new_len, k:k + new_len] = X[y]
                        y += 1
                        x += 1
        output_array.append(output)
    return np.array(output_array)



def get_blob_based_perf(tp, fp, fn):
    print("TP : ", tp)
    print("FP : ", fp)
    print("FN : ", fn)
    if tp == 0 and fp == 0 and fn == 0:
        dice = 1
    else:
        dice = tp / (0.00001 + tp + 0.5 * (fp + fn))

    print("Blob Dice : ", dice)

def tiffs_to_nifti(stain_dir='iba1_orig/annotated/'):
    for patches in os.listdir(stain_dir):
        pn = patches.split(' ')[-1]
        tiff_stack = []
        tiff_files =  sorted([t for t in os.listdir(stain_dir + patches + '/mask/') if t.endswith('.tiff')])
        #tiff_files = tiff_files.sort()
        for f in tiff_files:
            tiff_stack.append(np.array(Image.open(stain_dir + patches + '/mask/' + f)))
        nifti_vol = np.stack(tiff_stack)
        nifti_vol = np.swapaxes(nifti_vol, 0, 2)
        nifti_vol = np.rot90(nifti_vol, k=1, axes=(0, 1))
        write_nifti('iba1_orig/' + 'gt/patchvolume_' + str(pn) + '.nii.gz', nifti_vol)

def save_empty_gt(patch_nums, cube_size=200, gt_path='iba1_empty/gt/'):
    for pn in patch_nums:
        gt = np.zeros((cube_size, cube_size, cube_size))
        write_nifti(gt_path + 'patchvolume_' + str(pn) + '.nii.gz', gt)

def nearest_soma_encoding(soma_blobs_path, blobs_csv_path, soma_nn_path):
    ctr = 0
    for soma_pred_fname in os.listdir(soma_blobs_path):
        pn = soma_pred_fname.split('_')[1].split('.')[0]
        ctr += 1
        print(ctr, soma_pred_fname)
        soma_pred_vol = read_nifti(soma_blobs_path + soma_pred_fname)
        try:
            df = pd.read_csv(blobs_csv_path + 'patchvolume_'+str(pn)+'_dist.csv', index_col=0)
            points_list =  list(df['points'])
            brain_nn_dist_list = list(df['brain_nearest_dist'])

            for i in range(len(points_list)):
                points = [re.findall('\d+', key) for key in re.findall('\[\d+, \d+, \d+\]', points_list[i])]
                for p in points:
                    soma_pred_vol[int(p[0]), int(p[1]), int(p[2])] = np.round(float(brain_nn_dist_list[i]), 3)
        except Exception as e:
            pass

        write_nifti(soma_nn_path + soma_pred_fname, soma_pred_vol)

def rotate_nifti(vol):
    print(vol)

if __name__ == "__main__":
    #a='/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/full_nn_blobs_csv/'
    #for f in os.listdir(a):
    #    df = pd.read_csv(a+f, index_col=0)
    #    print(df.columns)
    src_dir = 'iba1_orig/gt/'
    for f in os.listdir(src_dir):
        #img = nib.load(src_dir+f)
        #print(img.header)
        vol = read_nifti(src_dir+f)
        vol = np.swapaxes(vol, 0, 2)
        vol = np.rot90(vol, k=1, axes=(0, 1))
        write_nifti('iba1_orig/gt02/'+f, vol)

    #nearest_soma_encoding('/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/somata_pred/',
    #                      '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/full_nn_blobs_csv/',
    #                      '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/somata_dist_blobs/')


    #save_empty_gt(list(range(500,  511)))
    #new_patches = ['2142', '2147', '2656',  '3897', '7961', '0', '1', '2', '3', '4']
    #save_empty_gt(new_patches)
    #new_patches = ['2142', '2147', '2656', '3666' , '3897', '4594', '5126', '5483', '12081']

    tiffs_to_nifti()




    patches = [x.split('.')[0].split('_')[1] for x in os.listdir(raw_files_path)]
    patch_num = patches[3]
    X = read_nifti(raw_files_path + 'patchvolume_' + str(patch_num) + '.nii.gz')

    X_arr = np.array(get_3D_slices(X, old_len, new_len))