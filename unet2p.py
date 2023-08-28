import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

def unet2p(pretrained_weights = None, input_size = (256,256,3)) :
    filters = [64, 128, 256, 512, 1024]

    inp = Input(input_size)
    #downsampling
    conv00 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv00 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv00)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv00)
    conv10 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv10 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv10)
    conv20 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv20 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv20)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv20)
    conv30 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv30 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv30)
    drop4 = Dropout(0.5)(conv30)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv40 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv40 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv40)
    drop5 = Dropout(0.5)(conv40)

    #upsampling
    ccat01 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv10)
    ccat01 = concatenate([ccat01, conv00])
    conv01 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat01)
    conv01 = Dropout(0.2)(conv01)
    conv01 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv01)

    ccat11 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv20)
    ccat11 = concatenate([ccat11, conv10])
    conv11 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat11)
    conv11 = Dropout(0.2)(conv11)
    conv11 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv11)

    ccat21 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(conv30)
    ccat21 = concatenate([ccat21, conv20])
    conv21 = Conv2D(filters[2], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat21)
    conv21 = Dropout(0.2)(conv21)
    conv21 = Conv2D(filters[2], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv21)

    ccat31 = Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same')(conv40)
    ccat31 = concatenate([ccat31, conv30])
    conv31 = Conv2D(filters[3], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat31)
    conv31 = Dropout(0.2)(conv31)
    conv31 = Conv2D(filters[3], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv31)

    ccat02 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv11)
    ccat02 = concatenate([ccat02, conv00, conv01])
    conv02 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat02)
    conv02 = Dropout(0.2)(conv02)
    conv02 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv02)

    ccat12 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv21)
    ccat12 = concatenate([ccat12, conv10, conv11])
    conv12 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat12)
    conv12 = Dropout(0.2)(conv12)
    conv12 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv12)

    ccat22 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(conv31)
    ccat22 = concatenate([ccat22, conv20, conv21])
    conv22 = Conv2D(filters[2], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat22)
    conv22 = Dropout(0.2)(conv22)
    conv22 = Conv2D(filters[2], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv22)

    ccat03 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv12)
    ccat03 = concatenate([ccat03, conv00, conv01, conv02])
    conv03 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat03)
    conv03 = Dropout(0.2)(conv03)
    conv03 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv03)

    ccat13 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv22)
    ccat13 = concatenate([ccat13, conv10, conv11, conv12])
    conv13 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat13)
    conv13 = Dropout(0.2)(conv13)
    conv13 = Conv2D(filters[1], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv13)

    ccat04 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv13)
    ccat04 = concatenate([ccat04, conv00, conv01, conv02, conv03])
    conv04 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(ccat04)
    conv04 = Dropout(0.2)(conv04)
    conv04 = Conv2D(filters[0], (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv04)

    final01 = Conv2D(7, 1, activation='elu')(conv01)
    final02 = Conv2D(7, 1, activation='elu')(conv02)
    final03 = Conv2D(7, 1, activation='elu')(conv03)
    final04 = Conv2D(7, 1, activation='elu')(conv04)

    final = (final01 + final02 + final03 + final04) / 4

    f = Conv2D(7, 1, activation='sigmoid')(final)

    model = Model(inputs=inp, outputs=f)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])



    return model