import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_model_loss_plots(best_history, plot_fname='losses.png'):

    fig, axes = plt.subplots(1,2, sharex=True, figsize=(10,8))

    axes[0].plot(best_history.history['loss'])
    axes[0].plot(best_history.history['val_loss'])
    axes[0].set_title('Model loss')
    axes[0].set_ylabel('Model loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Val'], loc='upper left')
    axes[0].grid(True)

    axes[1].plot(best_history.history['acc'])
    axes[1].plot(best_history.history['val_acc'])
    axes[1].set_title('Model accuracy')
    axes[1].set_ylabel('BCE acc')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Val'], loc='upper left')
    axes[1].grid(True)

    fig.suptitle('Model performance')
    plt.savefig(plot_fname)
    plt.close()

def get_2D_slices_plots(test_patch_seq, X_test, y_test, test_preds):
    fig, axes = plt.subplots(3, 5)
    random_slices = list(np.random.choice(200, 5))
    test_slice = np.random.choice(3, 1)[0]
    slice_num = test_patch_seq[test_slice]

    for j in range(len(random_slices)):
        axes[0][j].imshow(X_test[test_slice, :, :, random_slices[j]], cmap='gray')
        axes[0][j].set_axis_off()

        axes[1][j].imshow(y_test[test_slice, :, :, random_slices[j]], cmap='gray')
        axes[1][j].set_axis_off()

        axes[2][j].imshow(test_preds[test_slice, :, :, random_slices[j]], cmap='gray')
        axes[2][j].set_axis_off()

    axes[0][0].set_ylabel('raw')
    axes[1][0].set_ylabel('gt')
    axes[2][0].set_ylabel('pred')
    fig.suptitle('2D Slice for patch num ' + str(slice_num))
    plt.savefig(str(slice_num) + '_slices.png')
    plt.close()



def get_over_under_segmenting_plots(test_patch_seq, y_test, test_preds, plot_filename='somata_percent.png'):
    ones_percent = {}
    for k in range(len(test_patch_seq)):
        ones_percent[test_patch_seq[k]] = {}

        if len(np.unique(y_test[k], return_counts=True)[1]) > 1:
            ones_percent[test_patch_seq[k]]['actual'] = np.unique(y_test[k], return_counts=True)[1][1] / np.size(y_test[k])
        else:
            ones_percent[test_patch_seq[k]]['actual'] = 0 / np.size(y_test[k])

        if len(np.unique(test_preds[k], return_counts=True)[1]) > 1:
            ones_percent[test_patch_seq[k]]['pred'] = np.unique(test_preds[k], return_counts=True)[1][1] / np.size(y_test[k])
        else:
            ones_percent[test_patch_seq[k]]['pred'] = 0 / np.size(y_test[k])

    pd.DataFrame(ones_percent).T.plot(kind='bar')
    plt.xlabel('patch num')
    plt.ylabel('percentage')
    plt.title('What percent of 3D scan is somata?')
    plt.savefig(plot_filename)
    plt.close()
