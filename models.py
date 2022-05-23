from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, GlobalAveragePooling3D, Dropout, concatenate, \
    Conv2D, Conv3D, MaxPooling3D, Conv3DTranspose, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, ZeroPadding3D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l1, l2
from metrics import dice_coef, dice_coef_loss


def custom_unet2(img_width=200, img_height=200, img_depth=200):
    inputs = Input((img_width, img_height, img_depth, 1))

    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', )(inputs)
                   #kernel_regularizer=l1(1e-5),
                   #kernel_initializer='glorot_uniform')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', )(pool1)
    # kernel_regularizer=l1(1e-5),
    # kernel_initializer='glorot_uniform')(inputs)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)

    convT = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)#,
                            #kernel_regularizer=l2(1e-6),
                            #kernel_inkernel_regularizer=l1(1e-3)itializer='glorot_uniform')(pool1)
    if convT.shape[1] + 1 == conv3.shape[1]:
        #Add ZeroPadding3D
        convT = ZeroPadding3D(padding=((1,0), (1,0), (1,0)))(convT)

    up6 = concatenate([convT, conv3], axis=4)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)

    up7 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)

    up8 = concatenate([Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up8)
    conv9 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)#,
                    #kernel_regularizer=l1(1e-3),
                    #kernel_initializer='glorot_uniform')(up6)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model


def custom_unet(img_width=200, img_height=200, img_depth=200):
    inputs = Input((img_width, img_height, img_depth, 1))

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', )(inputs)
                   #kernel_regularizer=l1(1e-5),
                   #kernel_initializer='glorot_uniform')(inputs)
    #conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    #conv2 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool1)
    #conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    convT = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_regularizer=l1(1e-8))(pool1)#,
                            #kernel_regularizer=l2(1e-6),
                            #kernel_inkernel_regularizer=l1(1e-3)itializer='glorot_uniform')(pool1)
    if convT.shape[1] + 1 == conv1.shape[1]:
        #Add ZeroPadding3D
        convT = ZeroPadding3D(padding=((1,0), (1,0), (1,0)))(convT)

    up6 = concatenate([convT, conv1], axis=4)
    #conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(up6)#,
                    #kernel_regularizer=l1(1e-3),
                    #kernel_initializer='glorot_uniform')(up6)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def modify_2Dunet(img_width=200, img_height=200, img_depth=200):
    inputs = Input((img_width, img_height, img_depth, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    convT = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    if convT.shape[1] + 1 == conv4.shape[1]:
        #Add ZeroPadding3D
        convT = ZeroPadding3D(padding=((1,0), (1,0), (1,0)))(convT)

    up6 = concatenate([convT, conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    convT2 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    if convT2.shape[1] + 1 == conv3.shape[1]:
        #Add ZeroPadding3D
        convT2 = ZeroPadding3D(padding=((1,0), (1,0), (1,0)))(convT2)
    up7 = concatenate([convT2, conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    print(model.summary())
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

    #model.compile(optimizer=Adam(lr=1e-4, decay=1e-7),
    #              loss=["binary_crossentropy"],
    #              metrics=["acc"])#, f1_score, sensitivity, specificity])#, options=run_opts)

    return model
