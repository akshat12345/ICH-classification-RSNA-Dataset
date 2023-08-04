import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,3"

import pickle
import numpy as np

# import matplotlib.pyplot as plt

import math
from tensorflow.keras import metrics
# import keras

# from keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import initializers,regularizers
# from tensorflow.keras.optimizers import schedules

# import time

from keras import backend as K
from keras.models import Model, load_model#,Sequential
from keras.layers import Input, BatchNormalization, Dense, Flatten,Dropout,LeakyReLU ,Lambda#Activation,, Add
# from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D#, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D#, GlobalMaxPool3D, MaxPooling2D, MaxPool3D
# from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,CSVLogger
from keras.optimizers import SGD,Adam,RMSprop
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import regularizers
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,classification_report
from sklearn.utils import class_weight


def lrn3D(x):
	tensor = tf.unstack(x,axis=-1)
	res = [tf.nn.local_response_normalization(t) for t in tensor]
	return tf.stack(res,axis = -1)

def get_model_r(input_img): 
	wtinit = initializers.RandomNormal(mean=0.0, stddev=1.0)
	# print('\t--inpc',K.int_shape(input_img))	
	conv_layer1 = Conv3D(filters=48, kernel_size=3,strides = (2,2,1), activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(input_img)
	batchnorm1 = BatchNormalization()(conv_layer1)
	print('\t--1C',K.int_shape(conv_layer1))
	# batchnorm1 = Lambda(lrn3D)(conv_layer1)
	pooling_layer1 = MaxPooling3D(pool_size=(2,2,1),strides = (2,2,1), data_format = 'channels_last',)(batchnorm1) #8/2 = 4 output
	print('\t--2p',K.int_shape(pooling_layer1))
	drp0 = Dropout(0.1)(pooling_layer1)

	conv_layer2 = Conv3D(filters=64, kernel_size=3,strides = (2,2,1),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(drp0)
	batchnorm2 = BatchNormalization()(conv_layer2)
	print('\t--3c',K.int_shape(conv_layer2))
	# batchnorm2 = Lambda(lrn3D)(conv_layer2)
	pooling_layer2 = MaxPooling3D(pool_size=(2,2,1),strides = (2,2,1), data_format = 'channels_last',)(batchnorm2) #4/2 = 2 output
	print('\t--4p',K.int_shape(pooling_layer2))
	drp1 = Dropout(0.1)(pooling_layer2)


	conv_layer3 = Conv3D(filters=96, kernel_size=3, strides = (2,2,1),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(drp1)
	# batchnorm3 = BatchNormalization()(conv_layer3)
	print('\t--5c',K.int_shape(conv_layer3))
	conv_layer4 = Conv3D(filters=128, kernel_size=3, strides = (2,2,1),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(conv_layer3) #conv_layer3#batchnorm3
	# batchnorm4 = BatchNormalization()(conv_layer4)
	print('\t--6c',K.int_shape(conv_layer4))
	conv_layer5 = Conv3D(filters=256, kernel_size=3, strides = (1,1,1),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(conv_layer4) #conv_layer4#batchnorm4
	batchnorm5 = BatchNormalization()(conv_layer5)
	print('\t--7c',K.int_shape(conv_layer5))


	pooling_layer3 = MaxPooling3D(pool_size=(2,2,2),strides = (2,2,2), data_format = 'channels_last',)(batchnorm5)#conv_layer5# batchnorm5#2/2 = 1 output
	print('\t--8p',K.int_shape(pooling_layer3))
	drp2 = Dropout(0.1)(pooling_layer3)

	flatten_layer = Flatten()(drp2)
	dense_layer1 = Dense(units=1000, activation='relu')(flatten_layer)
	

	# drp3 = Dropout(0.1)(dense_layer1)	
	
	dense_layer2 = Dense(units=1000, activation='relu')(dense_layer1)
	
	output_layer = Dense(units=3, activation='softmax')(dense_layer2)
	model = Model(inputs=input_img, outputs=output_layer)
	return model





input_img = Input((256,256, 3, 1), name='img')#Input((512,512, 24, 1), name='img')
model = get_model_r(input_img)
model.summary()