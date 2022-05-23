import os
import datetime
import random
import numpy as np
from plots import get_over_under_segmenting_plots, get_2D_slices_plots, get_model_loss_plots
from utils import minmax_normalize, create_overlay, get_blob_based_perf, get_spliced_arrays, reshape_spliced_arrays
from read_img import read_nifti, write_nifti
from models import custom_unet, custom_unet2, modify_2Dunet
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, GlobalAveragePooling3D, Dropout, concatenate, \
    Conv2D, Conv3D, MaxPooling3D, Conv3DTranspose, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, ZeroPadding3D
from keras.optimizers import Adam, SGD
from metrics import dice_coef, f1_score, specificity, sensitivity, accuracy, dice_coef_loss
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from blobanalysis import get_patch_overlap, get_blobs_dist_vol

np.random.seed(5)
tf.compat.v1.set_random_seed(5)
from tensorboard.plugins.hparams import api as hp

config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1, 'CPU':10})
#gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

raw_files_path = 'combined/raw/'
gt_files_path = 'combined/gt/'


num_epochs = 500
batch_size = 1
orig_size = 200

if __name__ == "__main__":
    patch_data = {}

    patches = [x.split('.')[0].split('_')[1] for x in os.listdir(raw_files_path)]

    for patch_num in patches:
        patch_data[patch_num] = {}
        try:
            patch_data[patch_num]['raw'] = read_nifti(raw_files_path + 'patchvolume_' + str(patch_num) + '.nii')
        except Exception as e:
            patch_data[patch_num]['raw']    = read_nifti(raw_files_path + 'patchvolume_' + str(patch_num) + '.nii.gz')
        patch_data[patch_num]['gt']     = read_nifti(gt_files_path + 'patchvolume_' + str(patch_num) + '.nii.gz')

        #clip ground truth from different somata number values to just one
        patch_data[patch_num]['gt'][patch_data[patch_num]['gt'] > 0] = 1

    #generate random 20% patches for test set and the rest for training
    test_set_ratio = 0.3


    #k-fold validation 5 times, each time store performances of train, val, test sets on each patch number
    patch_fold_metrics = []
    overall_folds_metrics = []

    test_patch_seq = random.sample(patches, 6)




    chunk_perf = {}     #model performance for different
    for splice_by in [2]:#[2, 5, 10, 20, 40, 100]:
        best_test_dice_coef = 0
        best_wts_file = ""
        best_history = ""
        for i in range(5):
            training_patches = [p for p in patches if p not in test_patch_seq]

            val_patch_seq = random.sample(training_patches, 8)
            train_patch_seq = [p for p in training_patches if p not in val_patch_seq]

            X_train = [patch_data[pn]['raw'] for pn in train_patch_seq]
            y_train = [patch_data[pn]['gt'] for pn in train_patch_seq]

            X_val = [patch_data[pn]['raw'] for pn in val_patch_seq]
            y_val = [patch_data[pn]['gt'] for pn in val_patch_seq]

            X_test = [patch_data[pn]['raw'] for pn in test_patch_seq]
            y_test = [patch_data[pn]['gt'] for pn in test_patch_seq]

            print("X_train shape : ", np.shape(X_train), " y_train shape : ", np.shape(y_train))
            print("X_val shape : ", np.shape(X_val), " y_val shape : ", np.shape(y_val))
            print("X_test shape : ", np.shape(X_test), " y_test shape : ", np.shape(y_test))


            #normalize X dataset mean/std
            X_train = [minmax_normalize(arr) for arr in X_train]
            X_test = [minmax_normalize(arr) for arr in X_test]
            X_val = [minmax_normalize(arr) for arr in X_val]
            #make val set
            #fit generator
            #round numpy arrays to 4 digits

            X_train = np.round(np.array(X_train, dtype='float32'), 4)
            X_val = np.round(np.array(X_val, dtype='float32'), 4)
            y_val = np.array(y_val, dtype='float32')
            y_train = np.array(y_train, dtype='float32')

            X_test = np.round(np.array(X_test, dtype='float32'), 4)
            y_test = np.array(y_test, dtype='float32')

            print(X_train.shape)



            #Dvide 200 size cube into 200/4 = 50 len cubes
            X_train_spliced, y_train_spliced = get_spliced_arrays(X_train, y_train, splice_by=splice_by, normalize="None")
            X_val_spliced, y_val_spliced = get_spliced_arrays(X_val, y_val, splice_by=splice_by, normalize="None")
            X_test_spliced, y_test_spliced = get_spliced_arrays(X_test, y_test, splice_by=splice_by, normalize="None")

            print(X_train_spliced.shape, y_train_spliced.shape)
            print(X_val_spliced.shape, y_val_spliced.shape)
            print(X_test_spliced.shape, y_test_spliced.shape)


            X_train = X_train_spliced
            y_train = y_train_spliced
            X_val = X_val_spliced
            y_val = y_val_spliced
            X_test = X_test_spliced
            y_test = y_test_spliced
            # Prepare the training dataset.
            def get_dataset(X, y, shuffle=False):
                dataset = tf.data.Dataset.from_tensor_slices((X, y))
                #train_dataset = train_dataset.shuffle(buffer_size=1).map(train_preprocessing).batch(batch_size).prefetch(1)
                if shuffle:
                    dataset = dataset.shuffle(buffer_size=1).batch(batch_size).prefetch(1)
                else:
                    dataset = dataset.batch(batch_size).prefetch(1)
                return dataset

            train_dataset = get_dataset(X_train, y_train, shuffle=True)
            val_dataset = get_dataset(X_val, y_val)
            test_dataset = get_dataset(X_test, y_test)


            model = modify_2Dunet(img_width=X_train.shape[-3], img_depth=X_train.shape[-2], img_height=X_train.shape[-1])

            print(model.summary())
            run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

            model.compile(optimizer=Adam(lr=1e-4, decay=1e-7),
                          loss=[dice_coef_loss],  # "binary_crossentropy"],
                          metrics=["acc", dice_coef])  # , f1_score, sensitivity, specificity])#, options=run_opts)


            #model = modify_2Dunet()
            wts_file = 'weights_fold_'+str(i)+'.h5'
            model_checkpoint = ModelCheckpoint(wts_file, monitor='val_loss', save_best_only=True)
            # Saving the weights and the loss of the best predictions we obtained

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_fold_" + str(i))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            early_stopping_callback = EarlyStopping(patience=5)
            print('-' * 30)
            print('Fitting model...')
            print('-' * 30)
            history = model.fit(train_dataset, epochs=num_epochs, verbose=1, #shuffle=True,
                                validation_data=val_dataset,
                                #batch_size=1,
                                #validation_split=0.2,
                                callbacks=[model_checkpoint, tensorboard_callback, early_stopping_callback])

            print('-' * 30)
            print('Loading saved weights...')
            print('-' * 30)
            model.load_weights(wts_file)

            print('-' * 30)
            print('Predicting masks on test data...')
            print('-' * 30)

            print("Fold number : " + str(i))

            test_preds = model.predict(X_test, verbose=1, batch_size=batch_size)
            val_preds = model.predict(X_val, verbose=1, batch_size=batch_size)
            train_preds = model.predict(X_train, verbose=1, batch_size=batch_size)

            '''#reshape the output back to original size of 200
            #test_preds = np.reshape(test_preds, (-1, orig_size, orig_size, orig_size, 1))
            test_preds = reshape_spliced_arrays(test_preds, pred=True)
            #val_preds = np.reshape(val_preds, (-1, orig_size, orig_size, orig_size, 1))
            val_preds = reshape_spliced_arrays(val_preds, pred=True)
            #train_preds = np.reshape(train_preds, (-1, orig_size, orig_size, orig_size, 1))
            train_preds = reshape_spliced_arrays(train_preds, pred=True)

            y_test_spliced_copy = y_test.copy()
            #y_train = np.reshape(y_train, (-1, orig_size, orig_size, orig_size))
            y_train = reshape_spliced_arrays(y_train, pred=False)
            #y_val = np.reshape(y_val, (-1, orig_size, orig_size, orig_size))
            y_val = reshape_spliced_arrays(y_val, pred=False)
            #y_test = np.reshape(y_test, (-1, orig_size, orig_size, orig_size))
            y_test = reshape_spliced_arrays(y_test, pred=False)'''
            fold_performance = {}

            def update_fold_perf(fold_performance, patch_num, y, y_pred, set):
                fold_performance[patch_num] = {}
                fold_performance[patch_num]['dice_coef'] = dice_coef(y, y_pred).numpy()
                # fold_performance[train_patch_seq[pi]]['blob_dice_coef'] = get_blob_dice(y_train[pi], np.reshape(train_preds[pi], (200, 200, 200)))
                # fold_performance[train_patch_seq[pi]]['f1_score'] = f1_score(y_train[pi], train_preds[pi]).numpy()
                fold_performance[patch_num]['sensitivity'] = sensitivity(y, y_pred).numpy()
                fold_performance[patch_num]['specificity'] = specificity(y, y_pred).numpy()
                fold_performance[patch_num]['accuracy'] = accuracy(y, y_pred).numpy()
                fold_performance[patch_num]['set'] = set
                return fold_performance

            for pi in range(len(train_patch_seq)):
                fold_performance = update_fold_perf(fold_performance, train_patch_seq[pi], y_train[pi], train_preds[pi], 'train')

            for pi in range(len(val_patch_seq)):
                fold_performance = update_fold_perf(fold_performance, val_patch_seq[pi], y_val[pi], val_preds[pi], 'val')

            for pi in range(len(test_patch_seq)):
                fold_performance = update_fold_perf(fold_performance, test_patch_seq[pi], y_test[pi], test_preds[pi], 'test')


            print(pd.DataFrame(fold_performance).T)
            patch_fold_metrics.append(fold_performance)

            overall_metrics = {}
            overall_metrics['train'] = {}
            overall_metrics['val'] = {}
            overall_metrics['test'] = {}

            overall_metrics['train']['dice_coef'] = dice_coef(y_train, train_preds).numpy()
            #overall_metrics['train']['blob_dice_coef'] = get_blob_dice(np.vstack([y_train[x] for x in range(len(y_train))]),
            #                                                           np.vstack([np.reshape(train_preds[x], (200, 200, 200)) for x in range(len(train_preds))]))

            overall_metrics['val']['dice_coef'] = dice_coef(y_val, val_preds).numpy()
            #overall_metrics['val']['blob_dice_coef'] = get_blob_dice(np.vstack([y_val[x] for x in range(len(y_val))]),
            #                                                           np.vstack(
            #                                                               [np.reshape(val_preds[x], (200, 200, 200)) for
            #                                                                x in range(len(val_preds))]))

            overall_metrics['test']['dice_coef'] = dice_coef(y_test, test_preds).numpy()

            #overall_metrics['train']['f1_score'] = f1_score(y_train, train_preds).numpy()
            #overall_metrics['val']['f1_score'] = f1_score(y_val, val_preds).numpy()
            #overall_metrics['test']['f1_score'] = f1_score(y_test, test_preds).numpy()

            overall_metrics['train']['sensitivity'] = sensitivity(y_train, train_preds).numpy()
            overall_metrics['val']['sensitivity'] = sensitivity(y_val, val_preds).numpy()
            overall_metrics['test']['sensitivity'] = sensitivity(y_test, test_preds).numpy()

            overall_metrics['train']['specificity'] = specificity(y_train, train_preds).numpy()
            overall_metrics['val']['specificity'] = specificity(y_val, val_preds).numpy()
            overall_metrics['test']['specificity'] = specificity(y_test, test_preds).numpy()

            overall_metrics['train']['accuracy'] = accuracy(y_train, train_preds).numpy()
            overall_metrics['val']['accuracy'] = accuracy(y_val, val_preds).numpy()
            overall_metrics['test']['accuracy'] = accuracy(y_test, test_preds).numpy()

            print("Overall Metrics")
            print(pd.DataFrame(overall_metrics).T)
            overall_folds_metrics.append(overall_metrics)

            if overall_metrics['val']['dice_coef'] > best_test_dice_coef:
                best_test_dice_coef = overall_metrics['val']['dice_coef']
                print("New best dice coef on the fold ", str(i))
                print("Best Dice Coef Now : ", str(best_test_dice_coef))
                best_wts_file = wts_file
                best_history = history

        #get accuracy metrics for all patch nums
        print(overall_folds_metrics)
        print(patch_fold_metrics)

        print("Best DICE Coef = ", str(best_test_dice_coef))
        best_fold = int(best_wts_file.split('.')[0][-1])
        print("Best Performances ")
        print(pd.DataFrame(overall_folds_metrics[best_fold]))

        chunk_perf[splice_by] = {}
        chunk_perf[splice_by]['patch'] = overall_folds_metrics[best_fold]

        # save loss functions
        get_model_loss_plots(best_history, plot_fname='losses_splice_by_' + str(splice_by) + '.png')

        model.load_weights(best_wts_file)
        model.save_weights('best_weights_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5")
        test_preds = model.predict(X_test, batch_size=batch_size)

        print(test_preds.shape)
        #test_preds = np.reshape(test_preds, (-1, orig_size, orig_size, orig_size, 1))
        #test_preds = reshape_spliced_arrays(test_preds)
        print(test_preds.shape)

        #F1 - score based on threshold values
        thresholds = list(np.arange(0.5, 1, 0.1))
        f1_scores_thresh = []
        best_score = 0
        best_thresh = 0
        for thresh in thresholds:
            score = f1_score(y_test, test_preds, thresh).numpy()
            if score > best_score:
                best_score = score
                best_thresh = thresh
            f1_scores_thresh.append(score)

        plt.plot(thresholds, f1_scores_thresh)
        plt.title('Test(3/20) F1-Score vs Sigmoid Threshold')
        plt.ylabel('F1_score')
        plt.grid(True)
        plt.savefig('f1_score_thresh.png')
        plt.close()

        #use best sigmoid threshold to change the output to 0 n 1
        test_preds = tf.cast(tf.greater(test_preds, best_thresh), tf.float32)

        test_set_tp, test_set_fp, test_set_fn = 0, 0, 0
        # et blob dice

        for i in range(len(y_test)):
            patch_num = test_patch_seq[int(i//8)]
            pred_patch = np.reshape(test_preds[i], test_preds[i].shape[:-1])

            try:
                tp, fp, fn, predicted_blobs = get_patch_overlap(pred_patch, y_test[i])

                print("Patch Num : ", patch_num, i%8)
                get_blob_based_perf(tp, fp, fn)

                test_set_tp += tp
                test_set_fp += fp
                test_set_fn += fn
            except Exception as e:
                continue

            #blobs_dist_df, blobs_dist_vol = get_blobs_dist_vol(predicted_blobs, pred_patch)

            #blobs_dist_df.to_csv('blobs_dist_csv/patchvolume_'+str(patch_num)+'_dist.csv')
            #write_nifti('blobs_dist_somata/patchvolume_' + str(patch_num) + '.nii.gz', blobs_dist_vol)



        print("Overall Test Set TP : ", test_set_tp)
        print("Overall Test Set FP : ", test_set_fp)
        print("Overall Test Set FN : ", test_set_fn)

        test_set_blob_dice = test_set_tp / (0.00001 + test_set_tp + 0.5 * (test_set_fp + test_set_fn))
        print("Overall Test Set Blob Dice : ", test_set_blob_dice)

        chunk_perf[splice_by]['blob_dice'] = test_set_blob_dice
        print(chunk_perf)
        with open("input_test.txt", "a") as f:
            f.write(str(chunk_perf) + '\n\n')
        f.close()

    print(chunk_perf)
    print(pd.DataFrame(chunk_perf))
    a = pd.DataFrame(chunk_perf)
    a.to_csv('input_splice_perf.csv')

    #save the test predictions to nii files
    new_len = 100
    for k in range(len(y_test)):
        pn = test_patch_seq[k//8]
        output_volume = np.reshape(test_preds[k], (new_len, new_len, new_len))
        write_nifti('output/patchvolume_'+str(pn)+'_' + str(k%8)+'.nii.gz', output_volume)

        #create overlay of predictions with gt
        overlay_volume = create_overlay(y_test[k], np.reshape(test_preds[k], (new_len, new_len, new_len)))
        write_nifti('overlays/patchvolume_'+str(pn)+'_' + str(k%8)+'.nii.gz', overlay_volume)


    #What percentage of voxel was predicted as somata
    get_over_under_segmenting_plots(test_patch_seq, y_test, test_preds)

    #Analyze 2D random slices
    get_2D_slices_plots(test_patch_seq, X_test, y_test, test_preds)
    #np.save('imgs_mask_test.npy', y_pred)
    #print('-' * 30)
    #print('Saving predicted masks to files...')
    #print('-' * 30)
    #pred_dir = 'preds'
    #if not os.path.exists(pred_dir):
    #    os.mkdir(pred_dir)

    #for k in range(len(imgs_mask_test)):
    #    a = rescale_intensity(imgs_test[k][:, :, 0], out_range=(-1, 1))
    #    b = (imgs_mask_test[k][:, :, 0]).astype('uint8')
    #    io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'), mark_boundaries(a, b))
    # Saving our predictions in the directory 'preds'



    # plotting our dice coeff results in function of the number of epochs

    #[np.unique(y, return_counts=True) for y in y_train]
    #print(patch_data)
