import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras import models
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def dhn(pretrained_weights = 'None',input_size = (256,256,1)):
#input
    inputs = Input(input_size)
#encoder
#layer1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_a')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_b')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#layer2
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_a')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_b')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#layer3
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_a')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3_b')(conv3)

    #CRB

    crb1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='crb1', dilation_rate=6)(conv3)
    crb2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='crb2', dilation_rate=12)(conv3)
    crb3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='crb3', dilation_rate=18)(conv3)
    crb4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='crb4', dilation_rate=24)(conv3)
    crb_merge= concatenate([crb1,crb2,crb3,crb4], axis = 3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(crb_merge)

#layer4
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_a')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_b')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#layer5
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5_a')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name="conv5_b")(conv5)
    drop5 = Dropout(0.5)(conv5)

#decoder
#layer6
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv6_a')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv6_b')(conv6)


#layer7
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7_a')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7_b')(conv7)

#layer8
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv8_a')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv8_b')(conv8)


#layer9
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name="conv9")(merge9)
    conv9_a = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name="conv9_a")(conv9)

#output1
    conv9_b = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_b")(conv9_a)
    output1 = Conv2D(1, 1, activation='sigmoid', name="final_layer")(conv9_b)

#output2
    up1_csl2 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    output2= Conv2D(1, 1, activation='sigmoid', name="layer2_output")(up1_csl2)

#output3
    up1_csl3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
    up2_csl3 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up1_csl3))
    output3= Conv2D(1, 1, activation='sigmoid', name="layer3_output")(up2_csl3)

#output4
    up1_csl4 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    up2_csl4 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up1_csl4))
    up3_csl4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up2_csl4))
    output4= Conv2D(1, 1, activation='sigmoid', name="layer4_output")(up3_csl4)

#output5
    up1_csl4 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up2_csl4 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up1_csl4))
    up3_csl4 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up2_csl4))
    up4_csl4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up3_csl4))
    output5= Conv2D(1, 1, activation='sigmoid', name="layer5_output")(up4_csl4)

#output6
    up1_csl6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up2_csl6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up1_csl6))
    up3_csl6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up2_csl6))
    output6= Conv2D(1, 1, activation='sigmoid', name="layer6_output")(up3_csl6)

#output7
    up1_csl7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up2_csl7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up1_csl7))
    output7= Conv2D(1, 1, activation='sigmoid', name="layer7_output")(up2_csl7)

#output8
    up1_csl8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    output8= Conv2D(1, 1, activation='sigmoid', name="layer8_output")(up1_csl8)

#output9
    output9 = Conv2D(1, 1, activation='sigmoid', name="layer9_output")(conv1)

    model = Model(input = inputs, output = output1)
    # model = Model(input = inputs, output = [predict1, predict2])
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer=SGD(lr=5e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # if(pretrained_weights):
    # 	model.load_weights('/home/SHussain/ULS/UNet/QHSP/layer5/model-epoch.1200-0.96.h5')

    return model

