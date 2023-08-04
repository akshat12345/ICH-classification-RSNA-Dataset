import os
os.environ["CUDA_VISIBLE_DEVICES"]= "6"


# from keras.optimizers.schedules import ExponentialDecay
# import tensorflow as tf
# # from tensorflow import keras
# from tensorflow.keras import initializers,regularizers
# from keras import backend as K
# from keras.models import Model, load_model#,Sequential
# from keras.layers import Input, BatchNormalization, Dense, Flatten,Dropout,LeakyReLU ,Lambda#Activation,, Add
# # from keras.layers.core import Lambda, RepeatVector, Reshape
# from keras.layers.convolutional import Conv3D#, Conv3DTranspose
# from keras.layers.pooling import MaxPooling3D#, GlobalMaxPool3D, MaxPooling2D, MaxPool3D
# # from keras.layers.merge import concatenate, add
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,CSVLogger
# from keras.optimizers import SGD,Adam,RMSprop
# # from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras import regularizers

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Dense
from tensorflow.keras.layers import AveragePooling3D, GlobalAveragePooling3D,MaxPooling3D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate, Add#,Flatten, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,CSVLogger
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,classification_report
from tensorflow.keras import regularizers

import glob
import numpy as np

from skimage.transform import resize
from sklearn.metrics import f1_score, roc_auc_score, log_loss
# import tensorflow as tf
# from keras import regularizers, initializers
# from keras.layers import BatchNormalization, Activation

# devices_names = ['GPU:2','GPU:3']

# strategy = tf.distribute.MirroredStrategy(devices=devices_names)

# with strategy.scope():

class HUNet:    
    def __init__(self,layerdepth,n_slicepersample):
        global layercount
        layercount = 1
    
        self.eps = 1.001e-5
        self.drpoutrate = 0.25
        self.compression=0.5
        self.grow_nb_filters = False #True #
        self.growth_rate = 4        
        self.n_blocks = [3,4,6,3]
        self.blk_filters = [64,128,128,256]#,32,64,128]     
        self.inputensor = Input((256,256, n_slicepersample, 1), name='img')
        self.n_classes = 6
        self.slicecount = n_slicepersample
        
        
        self.inputckernel = (7,7,7)     
        self.inputcstride = (1,1,1)
        self.inputxpoolstride = (2,2,1)
        self.inputxpoolkernel = (3,3,3)
        
        
            
        
                
    def createmodel(self):  
        global layercount
        if self.slicecount == 3:
            s1 = (2,2,1)
            s2 = (2,2,1)
        elif self.slicecount == 8:
            s1 = (2,2,1)
            s2 = (2,2,1)
        elif self.slicecount == 16:
            s1 = (2,2,2)
            s2 = (2,2,1)
        elif self.slicecount == 24:
            s1 = (2,2,2)
            s2 = (2,2,2)
        else:
            print('\nUnknown Slice Count')
        x = self.inputblock(self.inputensor,s1)

        x = self.inputblock(x,s2)

        x = self.createblocks(x)

        x = BatchNormalization(epsilon=self.eps)(x)
        x = ReLU()(x)

        x = GlobalAveragePooling3D()(x)
        # print('\nGAP',K.int_shape(x))
        
        
        output_layer = Dense(self.n_classes, activation = 'sigmoid')(x)
        # print('\nDense',K.int_shape(output_layer),layercount)
        
        
        model = Model(inputs=self.inputensor, outputs=output_layer)
        return model

    def inputblock(self,input_img,s):
        # print('\n\n ======================= Input Block =========================')
        global layercount

        a = Conv3D(64, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',kernel_regularizer = regularizers.l2(0.01),data_format = 'channels_last')(input_img)
        a = BatchNormalization(epsilon=self.eps)(a)
        a = ReLU()(a)
        # print('\nInp-Conv K1 S1',K.int_shape(a),layercount)       
        layercount+=1

        b = Conv3D(12, (3,3,3), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',kernel_regularizer = regularizers.l2(0.01),data_format = 'channels_last')(input_img)
        b = BatchNormalization(epsilon=self.eps)(b)
        b = ReLU()(b)
        # print('\nInp-Conv K3 S1',K.int_shape(b),layercount)       
        layercount+=1

        c = Conv3D(20, (3,3,3), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',kernel_regularizer = regularizers.l2(0.01),data_format = 'channels_last')(b)
        c = BatchNormalization(epsilon=self.eps)(c)
        c = ReLU()(c)
        # print('\nInp-Conv K3 S1',K.int_shape(c),layercount)       
        layercount+=1

        x = concatenate([c,b])

        d = Conv3D(32, (3,3,3), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',kernel_regularizer = regularizers.l2(0.01),data_format = 'channels_last')(c)
        d = BatchNormalization(epsilon=self.eps)(d)
        d = ReLU()(d)
        # print('\nInp-Conv K3 S1',K.int_shape(d),layercount)       
        layercount+=1

        x = concatenate([x,d])
        # print('\n\tInp-Conc',K.int_shape(x))  

        x = Add()([x,a])
        # print('\tAdd',K.int_shape(x))

        
        x = MaxPooling3D((3,3,3), strides = s, padding = 'same',data_format = 'channels_last')(x)
        # print('\nInp-Pool K3 S221',K.int_shape(x))

    
        return x
        
        
    

    def createblocks(self,x):

        for stage,rep in enumerate(self.n_blocks):              
             
            kernel = (3,3,3)  

            if stage == 0 : 
                if self.slicecount == 3:
                    pstride = (2,2,1)
                else:
                    pstride = (2,2,2)
                rfac = 2                
                # print('\n\n ======================= Res - ',stage,' =========================')
                for i in range(rep):
                    if i==0:
                        ispreact = False
                    else:
                        ispreact = True
                    x = self.rblock(x, self.blk_filters[stage],rfac, kernel, (1,1,1),ispreact)                  
                    
                    if i==0:
                        d = x                       
                    else:                       
                        d = concatenate([x,d])
                        # print('\tConcat',i,K.int_shape(d))            
                x = d
                # print('\tConcat Res',K.int_shape(x))
                # print('\n\t~~~~~~~~~~~~~~Bridge~~~~~~~~~~~~~~')               
                x = BatchNormalization(epsilon=self.eps)(x)
                x = ReLU()(x)
                # print('B-R')
                x = MaxPooling3D((3,3,3), strides = pstride, padding = 'same',data_format = 'channels_last')(x)
                # print('\nB-Pool K3 S222',K.int_shape(x))
                # x = Dropout(self.drpoutrate)(x)
                ispreact = False
                            
            if stage == 1:
                pstride = (2,2,1)
                rfac = 2                
                # print('\n\n ======================= Res - ',stage,' =========================')
                for i in range(rep):
                    if i==0:
                        ispreact = False
                    else:
                        ispreact = True
                    x = self.rblock(x, self.blk_filters[stage],rfac, kernel, (1,1,1),ispreact)                  
                    
                    if i==0:
                        d = x                       
                    else:
                        # print(K.int_shape(d))                     
                        d = concatenate([x,d])
                        # print('\tConcat',i,K.int_shape(d))            
                x = d
                # print('\n\t~~~~~~~~~~~~~~Bridge~~~~~~~~~~~~~~')
                # print('\tConcat Res',K.int_shape(x))
                x = BatchNormalization(epsilon=self.eps)(x)
                x = ReLU()(x)
                # print('B-R')
                x = MaxPooling3D((3,3,3), strides = pstride, padding = 'same',data_format = 'channels_last')(x)
                # print('\nInp-Pool K3 S221',K.int_shape(x))
                # x = Dropout(self.drpoutrate)(x)
            if stage == 2:              
                rfac = 5            
                # print('\n\n ======================= Res - ',stage,' =========================')
                for i in range(rep):
                    if i==0:
                        ispreact = False
                    else:
                        ispreact = True
                    x = self.rblock(x, self.blk_filters[stage],rfac, kernel, (1,1,1),ispreact)                  
                    
                    if i==0:
                        d = x                           
                    else:                       
                        d = concatenate([x,d])
                        # print('\tConcat',i,K.int_shape(d))            
                x = d
                # print('\tConcat Res',K.int_shape(x))

            if stage == 3:              
                rfac = 8            
                # print('\n\n ======================= Res - ',stage,' =========================')
                for i in range(rep):                    
                    x = self.rblock(x, self.blk_filters[stage],rfac, kernel, (1,1,1),ispreact)                                      
                    if i==0:
                        d = x                           
                    else:                       
                        d = concatenate([x,d])
                        # print('\tConcat',i,K.int_shape(d))            
                x = d
                # print('\tConcat Res',K.int_shape(x))
        return x

    

    def rblock(self,x, filters,rfac, kernel_size, stride,ispreact = True):
        global layercount
        if ispreact:
            x = BatchNormalization(epsilon=self.eps)(x)
            x = ReLU()(x)
            # print('B-R')

        if K.int_shape(x)[-1] != filters*rfac:
            shortcut = Conv3D(rfac*filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
            # print('\t Shortcut Conv K1 S1',K.int_shape(shortcut),layercount)
            layercount+=1
        else:
            shortcut = x
            # print('\t Shortcut identity',K.int_shape(shortcut))
            

        x = Conv3D(filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',use_bias=False,padding = 'same',data_format = 'channels_last')(x)
        # print('\t\t\t (1,1,1) -Conv, S (1,1,1)',K.int_shape(x),layercount)
        layercount+=1
        x = BatchNormalization(epsilon=self.eps)(x)
        x = ReLU()(x)

        x = Conv3D(filters, kernel_size, strides = stride, kernel_initializer = 'he_normal',use_bias=False,padding = 'same',data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(x)
        # print('\t\t\t',kernel_size,'-Conv, S',stride,K.int_shape(x),layercount)
        layercount+=1
        x = BatchNormalization(epsilon=self.eps)(x)
        x = ReLU()(x)

        x = Conv3D(rfac*filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
        # print('\t\t\t (1,1,1) -Conv, S (1,1,1)',K.int_shape(x),layercount)
        layercount+=1
        
        x = Add()([shortcut, x])
        # print('\tAdd',K.int_shape(x))
        
        return x


def data_generator(patients, batch_size):
    while True:
        np.random.shuffle(patients)
        for i in range(0, len(patients), batch_size):
            batch_patients = patients[i:i+batch_size]
            batch_x = []
            batch_y = []
            # sum = 0
            for patient_id in batch_patients:
                # data_file = os.path.join(data_dir + patients +  "_data.npy")
                # label_file = os.path.join(data_dir+ patients + "_label.npy")
                # data = []
                # label = []
                if os.path.isfile("./Normal/" + patient_id + '_data.npy'):
                    data_dir = "./Normal/"
                else:
                    data_dir = "./ICH/"

                # for data_dir in data_dirs :
                #     if os.path.isfile(data_dir + patient_id + '_data.npy'):
                # print(data_dir + patient_id + '_data.npy')
                imgs_out= np.load(data_dir + patient_id + '_data.npy').astype(np.float32)  # shape: (num_slices, 512, 512)
                # print('imgs_out',imgs_out.shape)
                imgs_out = np.moveaxis(imgs_out,2,1)
                imgs_out = np.moveaxis(imgs_out,2,3)
                # imgs_out = np.moveaxis(imgs_out,3,4)
                # print('imgs_out',imgs_out.shape)
                data = np.zeros((imgs_out.shape[0],256,256,3),dtype=np.float16)
                for n,i in enumerate(imgs_out):
                    data[n,:,:,:] = resize(imgs_out[n,:,:,:], data.shape[1:], anti_aliasing=True)
                label = np.load(data_dir + patient_id + '_label.npy')
                        # break

                # print(patient_id,' - ',data.shape)
                imgs_out= None

                batch_x.extend(data)
                batch_y.extend(label)

            #For 3D-BSPC Model
            # batch_x = np.moveaxis(batch_x,1,2)
            batch_x = np.expand_dims(np.array(batch_x),axis=-1)
            batch_y = np.array(batch_y)

            # print('batch_x',batch_x.shape)
            yield batch_x, batch_y


# keras = tf.keras
# Sequential = keras.Sequential
# Conv3D = keras.layers.Conv3D
# MaxPooling3D = keras.layers.MaxPooling3D
# Flatten = keras.layers.Flatten
# Dense = keras.layers.Dense
# Dropout = keras.layers.Dropout
# Adam = keras.optimizers.Adam

# input_shape = (3, 512, 512, 1)

# Define the batch size and number of epochs
batch_size = 1
epochs = 3000


train_ratio = .70
val_ratio = .15
test_ratio = .15
data_dirs = ["./Normal/", "./ICH/"]

train_patients = []
val_patients = []
test_patients = []

for data_dir in data_dirs:
    files = glob.glob(data_dir + r'*.npy')
    files = [val.split('/')[-1].split('\\')[-1].split('_data.npy')[0] for val in files if
             not val.endswith('_label.npy')]
    num_train = int(len(files) * train_ratio)
    num_val = int(len(files) * val_ratio)
    num_test = len(files) - num_train - num_val
    train_patients.extend(files[:num_train])
    val_patients.extend(files[num_train:num_train + num_val])
    test_patients.extend(files[num_train + num_val:])

# print(train_patients)
# print(test_patients)
# print(val_patients)
print(len(train_patients), len(val_patients), len(test_patients))

train_generator = data_generator(train_patients, batch_size)
val_generator = data_generator(val_patients, batch_size)
test_generator = data_generator(test_patients, batch_size)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# def calc_logloss(targets, outputs, eps=1e-5):
#     # for RSNA
#     try:
#         logloss_classes = [log_loss(np.floor(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
#     except ValueError as e:
#         logloss_classes = [1, 1, 1, 1, 1, 1]

#     return {
#         'logloss_classes': logloss_classes,
#         'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
#     }

# def get_model_r(input_img): 
#     wtinit = initializers.RandomNormal(mean=0.0, stddev=1.0)
#     # print('\t--inpc',K.int_shape(input_img))  
#     conv_layer1 = Conv3D(filters=48, kernel_size=3,strides = (1,2,2), activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(input_img)
#     batchnorm1 = BatchNormalization()(conv_layer1)
#     print('\t--1C',K.int_shape(conv_layer1))
#     # batchnorm1 = Lambda(lrn3D)(conv_layer1)
#     pooling_layer1 = MaxPooling3D(pool_size=(1,2,2),strides = (1,2,2), data_format = 'channels_last',)(batchnorm1) #8/2 = 4 output
#     print('\t--2p',K.int_shape(pooling_layer1))
#     drp0 = Dropout(0.1)(pooling_layer1)

#     conv_layer2 = Conv3D(filters=64, kernel_size=3,strides = (1,2,2),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(drp0)
#     batchnorm2 = BatchNormalization()(conv_layer2)
#     print('\t--3c',K.int_shape(conv_layer2))
#     # batchnorm2 = Lambda(lrn3D)(conv_layer2)
#     pooling_layer2 = MaxPooling3D(pool_size=(2,2,2),strides = (1,2,2), data_format = 'channels_last',)(batchnorm2) #4/2 = 2 output
#     print('\t--4p',K.int_shape(pooling_layer2))
#     drp1 = Dropout(0.1)(pooling_layer2)


#     conv_layer3 = Conv3D(filters=96, kernel_size=3, strides = (1,2,2),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(drp1)
#     # batchnorm3 = BatchNormalization()(conv_layer3)
#     print('\t--5c',K.int_shape(conv_layer3))
#     conv_layer4 = Conv3D(filters=128, kernel_size=3, strides = (1,2,2),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(conv_layer3) #conv_layer3#batchnorm3
#     # batchnorm4 = BatchNormalization()(conv_layer4)
#     print('\t--6c',K.int_shape(conv_layer4))
#     conv_layer5 = Conv3D(filters=256, kernel_size=3, strides = (1,1,1),activation='relu',kernel_initializer = wtinit, padding = 'same',  data_format = 'channels_last', kernel_regularizer = regularizers.l2(0.01))(conv_layer4) #conv_layer4#batchnorm4
#     batchnorm5 = BatchNormalization()(conv_layer5)
#     print('\t--7c',K.int_shape(conv_layer5))


#     pooling_layer3 = MaxPooling3D(pool_size=(2,2,2),strides = (1,2,2), data_format = 'channels_last',)(batchnorm5)#conv_layer5# batchnorm5#2/2 = 1 output
#     print('\t--8p',K.int_shape(pooling_layer3))
#     drp2 = Dropout(0.1)(pooling_layer3)

#     flatten_layer = Flatten()(drp2)
#     dense_layer1 = Dense(units=1000, activation='relu')(flatten_layer)
    

#     # drp3 = Dropout(0.1)(dense_layer1) 
    
#     dense_layer2 = Dense(units=1000, activation='relu')(dense_layer1)
    
#     output_layer = Dense(units=6, activation='sigmoid')(dense_layer2)
#     model = Model(inputs=input_img, outputs=output_layer)
#     return model

# # with strategy.scope():
# # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# # mc = ModelCheckpoint(MODELS_PATH + split_num +'_inter_' + ind + '_' + run_count +'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# # cv = CSVLogger(MODELS_PATH + split_num +'_inter_' + ind + '_' + run_count +'.csv',append=True)
# # # lrs = LearningRateScheduler(lr_step_decay, verbose=1)
# # rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=30, min_lr=0.00001)
# # optimizer=Adam(lr=1e-4, amsgrad=True)
# # optimizer=SGD(lr=1e-3) #,momentum=0.9
# # optimizer = RMSprop(lr=1e-4)


# #optimizer=Adam(lr=0.001, amsgrad=True)


# learning_rate = 0.001#0.001
# # # decay_rate = 1e-6
# momentum = 0.7#0.7
# # optimizer = SGD(lr=learning_rate, momentum=momentum)#, decay=decay_rate
# optimizer=Adam(lr=0.001, amsgrad=True)

# input_img = Input((3,256,256,1), name='img')#Input((512,512, 24, 1), name='img')
# model = get_model_r(input_img)
# model.summary()
# model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])#'categorical_crossentropy' #tversky_loss


# checkpoint_path = "cp.ckpt"
# cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 # save_weights_only=True,
                                                 # verbose=1)
# history = model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=len(train_patients) // batch_size + 1,
#     epochs=epochs,
#     verbose=1,
#     validation_data=val_generator,
#     validation_steps=len(val_patients) // batch_size + 1,
#     callbacks=[cp_callback]
# )

# model.save('Model-3D.h5')

# model.load_weights("cp.ckpt")
# score = model.evaluate_generator(
#     test_generator,
#     steps=len(test_patients) // batch_size,
#     verbose=1
# )

# print(score)
def log_loss(y_true, y_pred):
    return -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7), axis=-1)


class_weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weights, adjust as needed

def weighted_log_loss(y_true, y_pred):
    weights = tf.constant(class_weights, dtype=tf.float32)
    loss = -(weights * y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    return tf.reduce_sum(loss, axis=-1)



huobj = HUNet('60',3)
model = huobj.createmodel()
model.summary()
learning_rate = 0.001
momentum = 0.7
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('NCCTV-inter.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# cv = CSVLogger(self.MODELS_PATH + self.model_name +'_inter_' + self.ind + '_' + self.run_count +'.csv',append=True)
# lrs = LearningRateScheduler(self.lr_step_decay, verbose=1)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=20, min_lr=0.0001)

optimizer = SGD(lr=learning_rate, momentum=momentum)
# optimizer=Adam(lr=self.learning_rate, amsgrad=True)
model.compile(loss=log_loss,optimizer=optimizer,metrics=[f1])#'categorical_crossentropy' #tversky_loss


history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_patients) // batch_size + 1,
    epochs=epochs,
    verbose=1,
    validation_data=val_generator,
    validation_steps=len(val_patients) // batch_size + 1,
    callbacks=[es,mc,rlp]
)

model.save('NCCTV-final.h5')