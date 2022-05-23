from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy

np.random.seed(5)
tf.compat.v1.set_random_seed(5)
from tensorboard.plugins.hparams import api as hp

config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1, 'CPU':10})
#gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum(y_pred_f * (1 - y_true_f))
    fn = K.sum((1 - y_pred_f) * y_true_f)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + smooth)
    intersection = K.sum(y_true_f * y_pred_f)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    #macro_cost = tf.reduce_mean(cost)  # average on all labels
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    #return cost

def bce_plus_dice_loss(bce_wt):
  def bce_plus_dice(y_true, y_pred):
    return K.constant(bce_wt) * binary_crossentropy(y_true, y_pred) + K.constant(1-bce_wt) * (1+dice_coef_loss(y_true, y_pred))
  return bce_plus_dice



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_confusion_values(y_true, y_pred, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
    #tp = tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y_true), tf.float32)
    #fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true), axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true)), tf.float32)
    #fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true, axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true), tf.float32)
    tn = tf.cast(tf.math.count_nonzero((1 - y_pred) * (1-y_true)), tf.float32)

    return tp, fp, fn, tn


#true positivie rate : tp/tp+fn
def sensitivity(y_true, y_pred, thresh=0.5):
    tp, fp, fn, tn = get_confusion_values(y_true, y_pred, thresh=thresh)
    return tp/(tp+fn+1e-7)

#true positivie rate : tp/tp+fn
def accuracy(y_true, y_pred, thresh=0.5):
    tp, fp, fn, tn = get_confusion_values(y_true, y_pred, thresh=thresh)
    return (tp+tn)/(tp+fp+fn+tn+1e-7)


#true negative rate
def specificity(y_true, y_pred, thresh=0.5):
    tp, fp, fn, tn = get_confusion_values(y_true, y_pred, thresh=thresh)
    return tn/(tn+fp+1e-7)

#tf1 score
def f1_score(y_true, y_pred, thresh=0.5):
    tp, fp, fn, tn = get_confusion_values(y_true, y_pred, thresh=thresh)
    return 2*tp/(2*tp+ fn+fp+1e-7)

