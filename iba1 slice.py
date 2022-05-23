import re

import pandas as pd
import numpy as np
import os
from collections import Counter
import datetime
from utils import minmax_normalize
from blobanalysis import get_blobs, get_blobs_dist_vol
from read_img import read_nifti, write_nifti
from filehandling import pload
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, GlobalAveragePooling3D, Dropout, concatenate, \
    Conv2D, Conv3D, MaxPooling3D, Conv3DTranspose, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, ZeroPadding3D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy import ndimage
from blobanalysis import get_patch_overlap
from models import custom_unet


'''
range_plus = [-1, 0, 1]
steps = []
init_step = (4, 29, 3)
steps.append(init_step)
for x in range_plus:
    for y in range_plus:
        for z in range_plus:
            new_step = (init_step[0] + x, init_step[1] + y, init_step[2] + z)
            print(new_step)
            steps.append(new_step)
'''

#Counter({'Outside': 3400, 'Boundary': 317, 'Core': 6253, 'Candidate': 20})  cx3  : 9990

#Found Blobs in patches by model : {'Outside': 52, 'Boundary': 242, 'Core': 6215, 'Candidate': 19}

raw_files_path = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/Local/'
output_files_path = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/somata_pred/'
blobs_csv_path = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/blobs_csv/'
full_blobs_csv_path = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/full_nn_blobs_csv/'
blobs_somata_vol_path = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/somata_pred/'
pickle_file = '/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/Sync/region.pickledump'

orig_size = 200
wts_file = 'cx3_best.h5'
ctr = 0
batch_size = 20
best_threshold = 0.9

cxc_max_x = 54
cxc_max_y = 37
cxc_max_z = 5


pickle_region = pload(pickle_file)
id_patchstep_dict = {p['id']:p['patchstep'] for p in pickle_region['patches']}

cxc_arr = np.zeros((cxc_max_x,cxc_max_y,cxc_max_z))
for p in pickle_region['patches']:
    pstep = p['patchstep']
    cxc_arr[pstep[0], pstep[1], pstep[2]] = p['id']


#get nearest neighbor in nearby patches as well
for f in os.listdir(blobs_csv_path):
    pn = f.split('_')[1]
    curr_pstep = id_patchstep_dict[int(pn)]
    src_df = pd.read_csv(blobs_csv_path+f, index_col=0)

    neighboring_centroids = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                pstep_loc = [curr_pstep[0]+x, curr_pstep[1]+y, curr_pstep[2]+z]
                if (pstep_loc[0] < cxc_max_x and pstep_loc[0] >= 0) and (pstep_loc[1] < cxc_max_y and pstep_loc[1] >= 0) and (pstep_loc[2] < cxc_max_z and pstep_loc[2] >= 0):
                    print(pstep_loc)
                    neighboring_pn = cxc_arr[pstep_loc[0], pstep_loc[1], pstep_loc[2]]
                    print(neighboring_pn)

                    if int(neighboring_pn) != int(pn):
                        try:
                            dst_df = pd.read_csv(blobs_csv_path + 'patchvolume_' + str(int(neighboring_pn)) + '_dist.csv', index_col=0)
                            neighboring_centroids.extend(list(dst_df['offset_centroid']))

                        except Exception as e:
                            #csv file doesn't exist for this patch
                            continue

    print(neighboring_centroids)
    def get_brain_nn(x):
        curr_centroid = re.findall('\d+', x['offset_centroid'])
        curr_nn_dist = float(x['patch_nearest_dist'])

        for centroid in neighboring_centroids:
            centroid = re.findall('\d+', centroid)
            dist = np.sqrt((int(curr_centroid[0]) - int(centroid[0]))**2 + (int(curr_centroid[1]) - int(centroid[1]))**2 + (int(curr_centroid[2]) - int(centroid[2]))**2)
            if dist < curr_nn_dist:
                curr_nn_dist = dist
        return curr_nn_dist

    src_df['brain_nearest_dist'] = src_df.apply(get_brain_nn, axis=1)
    src_df.to_csv(full_blobs_csv_path + f)

print('done')




#1. for each csv file get patch num first
#2. then get it's patchstep
#3. iterate through the nearest 26 patch steps and open their csv and compute the new nearest distances
#3. a) if we find a new nearer blob.. update.. also store which patchvolume num it was from
#4. save these new distances in new folder.. nn with nearby patches





#stitch brain


stitch_vol = []
for f in os.listdir(blobs_somata_vol_path):
    for fl in range(5):
        x_layers = []
        for x_ctr in range(15):
            y_layers = []
            for y_ctr in range(37):
                z_layers = []
                for z_ctr in range(5):
                    vol = read_nifti(blobs_somata_vol_path + f)
                    z_layers.append(vol)
                    #print("z counter : ", z_ctr)
                concat_z_layer = np.concatenate(np.array(z_layers), axis=2)
                del z_layers
                #print(concat_z_layer.shape)
                y_layers.append(concat_z_layer)
                del concat_z_layer
                #print("y counter : ", y_ctr)
            concat_y_layer = np.concatenate(np.array(y_layers), axis=1)
            del y_layers
            #print(concat_y_layer.shape)
            x_layers.append(concat_y_layer)
            del concat_y_layer
            print("x counter : ", x_ctr)
        concat_x_layer = np.concatenate(np.array(x_layers), axis=0)
        print(concat_x_layer.shape)
        write_nifti('/media/harshsangwan/51A631B1183117DF/Microglia/cx3cr1gfp_145/Patches/model_brain_' + str(fl) + '.nii.gz', concat_x_layer)
'''    stitch_vol.append(concat_x_layer)
    

    full_brain = np.concatenate(np.array(stitch_vol))




if __name__ == "__main__":

    lens = []
    for f in os.listdir(blobs_csv_path):
        df = pd.read_csv(blobs_csv_path + f, index_col=0)
        print(len(df))
        if len(df) == 1:
            os.remove(blobs_csv_path + f)
            print('Removed ', f)
        else:
            lens.append(len(df))

        if len(df) > 400:
            print("AAAAAHHHHH")
            print("AAAAAHHHHH")
            print("AAAAAHHHHH")
            print(f)
            print("AAAAAHHHHH")
            print("AAAAAHHHHH")

    print(Counter(lens))

    #vol = read_nifti('cx3_empty_patches/gt/patchvolume_2.nii.gz')
    #a = get_blobs(vol)
    pickle_region = pload(pickle_file)
    offset_dict = {p['id']:p['offset'] for p in pickle_region['patches']}
    print(set([p['locationgroup'] for p in pickle_region['patches']]))


    patch_data = {}
    patches = [x.split('.')[0].split('_')[1] for x in os.listdir(raw_files_path)]

    patch_batches = [patches[i:i+batch_size] for i in range(0, len(patches), batch_size)]

    model = custom_unet()
    model.load_weights(wts_file)


    for i in range(len(patch_batches)):
        X_test = []
        patch_seq = []
        print("Batch Num : ", i)
        for pn in patch_batches[i]:
            patch_seq.append(pn)
            X_test.append(read_nifti(raw_files_path + 'patchvolume_' + str(pn) + '.nii.gz'))

        X_test = [minmax_normalize(arr) for arr in X_test]
        X_test = np.array(X_test, dtype='float32')
        X_test = np.round(X_test, 4)

        y_pred = model.predict(X_test, verbose=1, batch_size=1)
        y_pred = tf.cast(tf.greater(y_pred, best_threshold), tf.float32)

        for k in range(len(patch_seq)):
            
                print(offset_dict.keys())
                ctr = 0
                print('Total Patches : ', len(patches))
                for pn in patches:
            pn = patch_seq[k]
            #if int(pn) in offset_dict.keys():
            #output_volume = read_nifti(output_files_path + 'patchvolume_' + str(pn) + '.nii.gz')
            output_volume = np.reshape(y_pred[k], (orig_size, orig_size, orig_size))
            write_nifti(output_files_path + 'patchvolume_'+str(pn)+'.nii.gz', output_volume)
            print('Output patch somata : ', pn)

            predicted_blobs = get_blobs(output_volume)

            if predicted_blobs is None:
                pass
            else:
                patch_offset = offset_dict[int(pn)]
                print('patch offset ', patch_offset)

                blobs_dist_df, blobs_dist_vol = get_blobs_dist_vol(predicted_blobs, output_volume, patch_offset=patch_offset)

                if blobs_dist_df is not None:
                    print(blobs_dist_df)
                    blobs_dist_df.to_csv(blobs_csv_path + 'patchvolume_' + str(pn) + '_dist.csv')

            ctr += 1
            print('Finished patches : ', ctr)
                #write_nifti(blobs_somata_vol_path + 'patchvolume_' + str(pn) + '.nii.gz', blobs_dist_vol)
                    #add distance blob 3d vol output
                    #pred_vol = get_blobs(output_volume)

'''

