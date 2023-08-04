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
# from sklearn.utils import class_weight

import pickle
import os
import numpy as np
import time
import argparse
# import matplotlib.pyplot as plt
import math
import os
from datetime import datetime


class ResNetV1:	
	def __init__(self,layerdepth,n_slicepersample):
		global layercount
		layercount = 1
	
		self.eps = 1.001e-5
		self.drpoutrate = 0.4
		if layerdepth == '50':
			self.n_blocks = [3,4,6,3]
		elif layerdepth == '101':
			self.n_blocks = [3,4,23,3]
		elif layerdepth == '151':
			self.n_blocks = [3,8,36,3]
		else:
			print('\nUnknown ResNet Architecture Depth')
			return None

		self.blk_filters = [64,128,256,512]
		
		self.inputensor = Input((256,256, n_slicepersample, 1), name='img')
		self.n_classes = 3

		self.slicecount = n_slicepersample
		
		
		self.inputckernel = (7,7,7)
		if self.slicecount==3:
			self.inputxpoolstride = (2,2,1)
			self.inputcstride = (2,2,1)
			self.inputxpoolkernel = (3,3,3)
		elif self.slicecount==8:
			self.inputcstride = (2,2,1)
			self.inputxpoolstride = (2,2,1)#(2,2,2)
			self.inputxpoolkernel = (3,3,3)
		elif self.slicecount==16:
			self.inputcstride = (2,2,2)
			self.inputxpoolstride = (2,2,2)
			self.inputxpoolkernel = (3,3,3)
		elif self.slicecount==24:
			self.inputcstride = (2,2,2)
			self.inputxpoolstride = (2,2,2)
			self.inputxpoolkernel = (3,3,3)
		
	def setstridefor3D(self,i_conv,n_stage,slicecount,n_conv,n_filter):
		print('-------->>> ',n_stage,n_conv,i_conv,slicecount,n_filter)	
		ispool = False	
		isdropout = False
		# if i_conv != 0:
		cstrdestage = (1,1,1)
		# else:
		# 	cstrdestage = (2,2,2)

		if n_stage == 0:
			if slicecount==3:				
				# if i_conv != n_conv-1:
				# 	cstrdestage = (1,1,1)
				# else:
				# 	ispool = True
				cstrdestage = (1,1,1)
			elif slicecount==8:
				if i_conv == 0:
					cstrdestage = (2,2,2)#(2,2,1)
			elif slicecount==16:
				cstrdestage = (1,1,1)
			elif slicecount==24:
				cstrdestage = (1,1,1)			
			filters = n_filter[n_stage]			
		elif n_stage == 1:
			if slicecount==3:
				if i_conv == 0:
					cstrdestage = (2,2,1)
				# 	cstrdestage = (1,1,1)
				# else:
				# 	ispool = True
					cstrdestage = (2,2,1)
			elif slicecount==8:
				if i_conv == 0:
					cstrdestage = (2,2,1)
			elif slicecount==16:
				if i_conv == 0:
					cstrdestage = (2,2,1)
			elif slicecount==24:
				if i_conv == 0:
					cstrdestage = (2,2,1)					
			filters = n_filter[n_stage]
		elif n_stage == 2:
			if slicecount==3:
				if i_conv == 0:
					cstrdestage = (2,2,1)
			elif slicecount==8:
				# if i_conv == 0:
				cstrdestage = (1,1,1)
					# ispool = True
			elif slicecount==16:
				if i_conv == 0:
					cstrdestage = (2,2,1)
			elif slicecount==24:
				if i_conv == 0:
					cstrdestage = (2,2,2)
			filters = n_filter[n_stage]
		elif n_stage == 3:
			cstrdestage = (1,1,1)
			if slicecount==3:
				# if i_conv != n_conv-1:
				# 	cstrdestage = (1,1,1)
				# else:
				# 	ispool = True
				cstrdestage = (1,1,1)
			elif slicecount==8:
				# if i_conv == 0:
				cstrdestage = (1,1,1)
			elif slicecount==16:
				cstrdestage = (1,1,1)
			elif slicecount==24:
				cstrdestage = (1,1,1)
			filters = n_filter[n_stage]
		return cstrdestage,filters,ispool,isdropout
				
	def createmodelv1(self):	
		global layercount
		x = self.inputblockv1(self.inputensor,self.inputckernel,self.inputcstride,self.inputxpoolkernel,self.inputxpoolstride)

		x = self.resblockv1(x,self.n_blocks,self.blk_filters)

		x = GlobalAveragePooling3D()(x)
		print('\nGAP',K.int_shape(x),layercount)
		layercount+=1
		
		output_layer = Dense(self.n_classes, activation = 'softmax')(x)
		print('\nDense',K.int_shape(output_layer),layercount)
		
		model = Model(inputs=self.inputensor, outputs=output_layer)
		return model

	def inputblockv1(self,input_img,ckernel,cstride,xpoolk,xpstride):
		global layercount
		x = Conv3D(64, ckernel, strides = cstride, kernel_initializer = 'he_normal',padding = 'same',use_bias=True,data_format = 'channels_last')(input_img)
		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)
		print('\nInp-Conv',K.int_shape(x),layercount)		
		layercount+=1
		x = MaxPooling3D(xpoolk, strides = xpstride, padding = 'same',data_format = 'channels_last')(x)
		print('\nInp-Pool',K.int_shape(x),layercount)
		layercount+=1
		return x

	def resblockv1(self,x,blocks,n_filter):
		for stage,block in enumerate(blocks):
			print('\n\n ======================= Set - ',stage,' =========================')
			# cstride = (1,1,1)   
			# cstrdestage = (2,2,2)
			kernel = (3,3,3)   
			if stage == 0:				
				filters = n_filter[stage]
				cstrdestage = (1,1,1)
			elif stage == 1:
				filters = n_filter[stage]
			elif stage == 2:
				filters = n_filter[stage]
			elif stage == 3:
				filters = n_filter[stage]
			else:
				print('\n Only 4 Blocks Permissible in ResNet!!!')
				return None

			# x = self.commonblockv1(x, filters, kernel, cstride, block_type = 'Conv')
			# for _ in range(block-1):
			# 	x = self.commonblockv1(x, filters, kernel, cstride, block_type = 'Identity')





			for i in range(block):
				cstride,filters,ispool,isdropout = self.setstridefor3D(i,stage,self.slicecount,block,n_filter)
				if i != 0:
					# cstride = (1,1,1)
					print('\n\nStride ',cstride)
					x = self.commonblockv1(x, filters, kernel, cstride, block_type = 'Identity')
				else:					
					# cstride = cstrdestage
					print('\n\nStride ',cstride)
					x = self.commonblockv1(x, filters, kernel, cstride, block_type = 'Conv')					

		return x



	def commonblockv1(self,x, filters, kernel_size, stride, block_type):
		global layercount
		if block_type == 'Conv':			
			shortcut = Conv3D(4*filters, (1,1,1), strides = stride, kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
			print('\t\t\t (1,1,1) -Conv(shortcut)',K.int_shape(shortcut),4*filters)
			shortcut = BatchNormalization(epsilon=self.eps)(shortcut)	       
		else:
			shortcut = x
			

		x = Conv3D(filters, (1,1,1), strides = stride, kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
		print('\t\t\t (1,1,1) -Conv',K.int_shape(x),filters,layercount)
		layercount+=1
		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)

		x = Conv3D(filters, kernel_size, strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
		print('\t\t\t',kernel_size,'-Conv',K.int_shape(x),filters,layercount)
		layercount+=1
		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)

		x = Conv3D(4*filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
		print('\t\t\t (1,1,1) -Conv',K.int_shape(x),4*filters,layercount)
		layercount+=1
		x = BatchNormalization(epsilon=self.eps)(x)

		x = Add()([shortcut, x])
		print('\tAdd',K.int_shape(x))
		x = ReLU()(x)

		return x

class ResNetV2:
	def __init__(self,inptens,numclass,layerdepth):
		global layercount
		layercount = 1
		self.eps = 1.001e-5
		if layerdepth == '50':
			self.n_blocks = [3,4,6,3]
		elif layerdepth == '101':
			self.n_blocks = [3,4,23,3]
		elif layerdepth == '151':
			self.n_blocks = [3,8,36,3]
		else:
			print('\nUnknown ResNet Architecture Depth')
			return None

		self.blk_filters = [64,128,256,512]
		self.inputensor = inptens
		self.n_classes = numclass
		
		slicecount = K.int_shape(self.inputensor)[3]
		
		if slicecount==3 or slicecount==8 or slicecount==16:
			self.inputckernel = (7,7,7)
			if slicecount!=16:
				self.inputcstride = (2,2,1)
			else:
				self.inputcstride = (2,2,2)
			self.inputxpoolstride = (2,2,1)
			self.inputxpoolkernel = (3,3,3)
		else:
			self.inputckernel = (7,7,7)
			self.inputcstride = (2,2,2)
			self.inputxpoolstride = (2,2,2)
			self.inputxpoolkernel = (3,3,3)
		
				
	def createmodelv2(self):
		global layercount
		x = self.inputblockv2(self.inputensor,self.inputckernel,self.inputcstride,self.inputxpoolkernel,self.inputxpoolstride)

		x = self.resblockv2(x,self.n_blocks,self.blk_filters)

		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)

		x = GlobalAveragePooling3D()(x)
		print('\nGAP',K.int_shape(x),layercount)
		layercount+=1
		
		output_layer = Dense(self.n_classes, activation = 'softmax')(x)
		print('\nDense',K.int_shape(output_layer),layercount)
		layercount+=1
		model = Model(inputs=input_img, outputs=output_layer)
		return model

	def inputblockv2(self,input_img,ckernel,cstride,xpoolk,xpstride):
		global layercount
		x = Conv3D(64, ckernel, strides = cstride, kernel_initializer = 'he_normal',padding = 'same',use_bias=True,data_format = 'channels_last')(input_img)		
		print('\n',ckernel,'Inp-Conv, S',cstride,K.int_shape(x),layercount)
		layercount+=1
		x = MaxPooling3D(xpoolk, strides = xpstride, padding = 'same',data_format = 'channels_last')(x)
		print('\n',xpoolk,'Inp-Pool, S',xpstride,K.int_shape(x),layercount)
		layercount+=1
		return x

	def resblockv2(self,x,blocks,n_filter):		
		for stage,block in enumerate(blocks):
			print('\n\n ======================= Set - ',stage+1,' =========================')
			cstrdestage = (2,2,2)   
			kernel = (3,3,3)   
			if stage == 0:				
				filters = n_filter[stage]
			elif stage == 1:
				filters = n_filter[stage]
			elif stage == 2:
				filters = n_filter[stage]
			elif stage == 3:
				filters = n_filter[stage]
				cstrdestage = (1,1,1)
			else:
				print('\n Only 4 Blocks Permissible in ResNet!!!')
				return None
			
			
			for i in range(block):
				if i != block-1:
					cstride = (1,1,1)
				else:
					cstride = cstrdestage
				print('\n\nStride ',cstride)
				if i ==0:	
					x = self.commonblockv2(x, filters, kernel, cstride, block_type = 'Conv')	
				else:		
					x = self.commonblockv2(x, filters, kernel, cstride, block_type = 'Identity')

		return x

	def commonblockv2(self,x, filters, kernel_size, stride, block_type):
		global layercount
		preactivation = BatchNormalization(epsilon=self.eps)(x)
		preactivation = ReLU()(preactivation)

		if block_type == 'Conv':			
			shortcut = Conv3D(4*filters, (1,1,1), strides = stride, kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(preactivation)
			print('\t\t\t (1,1,1) -Conv(shortcut), S',stride,K.int_shape(shortcut),4*filters)			       
		else:
			if stride[0] >1 and stride[1] >1 and stride[2] >1:
				shortcut = MaxPooling3D((1,1,1), strides = stride, padding = 'same',data_format = 'channels_last')(x)	
				print('\t\t\t (1,1,1) -MxPool(shortcut), S',stride,K.int_shape(shortcut),4*filters)	
			else:
				shortcut = x
			

		x = Conv3D(filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',use_bias=False,padding = 'same',data_format = 'channels_last')(preactivation)
		print('\t\t\t (1,1,1) -Conv, S (1,1,1)',K.int_shape(x),filters,layercount)
		layercount+=1
		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)

		x = Conv3D(filters, kernel_size, strides = stride, kernel_initializer = 'he_normal',use_bias=False,padding = 'same',data_format = 'channels_last')(x)
		print('\t\t\t',kernel_size,'-Conv, S',stride,K.int_shape(x),filters,layercount)
		layercount+=1
		x = BatchNormalization(epsilon=self.eps)(x)
		x = ReLU()(x)

		x = Conv3D(4*filters, (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal',padding = 'same',data_format = 'channels_last')(x)
		print('\t\t\t (1,1,1) -Conv, S (1,1,1)',K.int_shape(x),4*filters,layercount)
		layercount+=1
		
		x = Add()([shortcut, x])
		print('\tAdd',K.int_shape(x))
		
		return x


class trainortest:	
	def __init__(self,gpuid,isTrain,dstype,slc,rcount,layerdepth):
		self.isTrain = isTrain
		self.final_model = '_final_'
		self.best_model = '_inter_'
		if slc == '16':
			slc = '1'
		elif slc == '24':
			slc = '2'
		self.dataset = slc+dstype+'1'
		# os.environ["CUDA_VISIBLE_DEVICES"]=gpuid
		self.ind = datetime.today().strftime('%Y-%m-%d')	#'2021-09-18' 	
		self.run_count = rcount
		self.model_name = self.dataset+'-resnetv1-'+layerdepth#+'NonSeg'
		self.MODELS_PATH = "/home/jenyrajan/NEETHI/DBT_Project/"
		self.learning_rate = 0.01
		self.momentum = 0.7
		self.num_epoch = 200
		self.batchsize = 32

	def tversky_loss(self,y_true, y_pred, alpha=0.15, beta=0.85, smooth=1e-10):
		""" Tversky loss function.
		Parameters
		----------
		y_true : keras tensor
			tensor containing target mask.
		y_pred : keras tensor
			tensor containing predicted mask.
		alpha : float
			real value, weight of '0' class.
		beta : float
			real value, weight of '1' class.
		smooth : float
			small real value used for avoiding division by zero error.
		Returns
		-------
		keras tensor
			tensor containing tversky loss.
		"""
		# y_pred = K.print_tensor(y_pred)
		y_true = K.flatten(y_true)
		y_pred = K.flatten(y_pred)
		truepos = K.sum(y_true * y_pred)
		fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
		answer = 1 - (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
		return answer

	def lr_step_decay(self,epoch, lr):
		# drop_rate = 0.2
		# epochs_drop = 35.0
		# newlr = lr
		if epoch==50:
			self.learning_rate = 0.001
		if epoch == 200:
			self.learning_rate = 0.0001		
		return self.learning_rate

	def f1(self,y_true, y_pred):
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

	def loadtesttrainevalddataserver(self,ind='Z'):
		name =''		
		if ind == 'Z':
			name += '_RZ'
		elif ind == 'M':		
			name += '_MaxNorm'
		print('\n\nFilename Loading Sample: XtRain-'+self.dataset+name+'.npy','-----','YtRain-'+self.dataset+name+'.npy','\n\n')
		try:		
			with open('XtRain-'+self.dataset+name+'.npy','rb') as f:
				xtr=np.load(f)#.astype(np.float16)  
			with open('XteSt-'+self.dataset+name+'.npy','rb') as f:
				xts=np.load(f)#.astype(np.float16)  
			with open('Xeval-'+self.dataset+name+'.npy','rb') as f:
				xev=np.load(f)#.astype(np.float16)
			with open('XteStCase-'+self.dataset+name+'.npy','rb') as f:
				xtscas=np.load(f,allow_pickle=True)#.astype(np.float16) 
			with open('YteStCase-'+self.dataset+name+'.npy','rb') as f:
				ytscase=np.load(f,allow_pickle=True)
			with open('Yeval-'+self.dataset+name+'.npy','rb') as f:
				yev=np.load(f)
			with open('YtRain-'+self.dataset+name+'.npy','rb') as f:
				ytr=np.load(f)
			with open('YteSt-'+self.dataset+name+'.npy','rb') as f:
				yts=np.load(f)
		except:
			print('No File Load From FinalDataSplit')
		# print('Check Norm/Std: ',xtr.max(),xts.max(),xev.max())
		return xtr,ytr,xts,yts,xev,yev,xtscas,ytscase

	def testevalddataload(self,ind='Z'):
		name =''		
		if ind == 'Z':
			name += '_RZ'
		elif ind == 'M':		
			name += '_MaxNorm'
		print('\n\nFilename Loading Sample: XteSt-'+self.dataset+name+'.npy','-----','YteSt-'+self.dataset+name+'.npy','\n\n')
		try:		
			
			with open('XteSt-'+self.dataset+name+'.npy','rb') as f:
				xts=np.load(f)#.astype(np.float16)  
			with open('Xeval-'+self.dataset+name+'.npy','rb') as f:
				xev=np.load(f)#.astype(np.float16)
			with open('XteStCase-'+self.dataset+name+'.npy','rb') as f:
				xtscas=np.load(f,allow_pickle=True)#.astype(np.float16) 
			with open('YteStCase-'+self.dataset+name+'.npy','rb') as f:
				ytscase=np.load(f,allow_pickle=True)
			with open('Yeval-'+self.dataset+name+'.npy','rb') as f:
				yev=np.load(f)			
			with open('YteSt-'+self.dataset+name+'.npy','rb') as f:
				yts=np.load(f)
		except:
			print('No File Load From FinalDataSplit')
		# print('Check Norm/Std: ',xtr.max(),xts.max(),xev.max())
		return xts,yts,xev,yev,xtscas,ytscase

	def startTrainingandTesting(self,model,m_test_typ='F'):	

		if self.isTrain:
			dloadstart = time.time()
			x_train,y_train,x_test,y_test,x_val,y_val,xtscas,ytscase  = self.loadtesttrainevalddataserver(ind='Z')
			print('\n\nData Shape : \n Train:',x_train.shape,y_train.shape,'\nVal:',x_val.shape,y_val.shape,'\nTest : ',x_test.shape,y_test.shape)
			dloadend = time.time()
			print('\nData Load Time : ',dloadend-dloadstart, ' seconds')

			trnstart = time.time()
			es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
			mc = ModelCheckpoint(self.MODELS_PATH + self.model_name +'_inter_' + self.ind + '_' + self.run_count +'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
			cv = CSVLogger(self.MODELS_PATH + self.model_name +'_inter_' + self.ind + '_' + self.run_count +'.csv',append=True)
			lrs = LearningRateScheduler(self.lr_step_decay, verbose=1)
			rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=15, min_lr=0.00001)
			
			optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)
			# optimizer=Adam(lr=self.learning_rate, amsgrad=True)

			model.compile(loss=self.tversky_loss,optimizer=optimizer,metrics=[self.f1])#'categorical_crossentropy' #tversky_loss
			
			results = model.fit(x_train, y_train, batch_size=self.batchsize , epochs=self.num_epoch, validation_data=(x_val, y_val), shuffle=True,callbacks=[mc,cv,es,rlp,lrs])
			model.save(self.MODELS_PATH + self.model_name +'_final_' + self.ind + '_' + self.run_count +'.h5')
			with open(self.MODELS_PATH + self.model_name +'_final_' + self.ind + '_' + self.run_count +'.pkl', 'wb') as fptr:
				pickle.dump(results.history, fptr)

			trnend = time.time()
			print('\nTraining Time : ',(trnend-trnstart)/60, ' minutes')
		else:			
			dtloadstart = time.time()
			x_test,y_test,x_val,y_val,xtscas,ytscase  = self.testevalddataload(ind='Z')
			print('\n\n Test Data Shape : \nVal:',x_val.shape,y_val.shape,'\nTest : ',x_test.shape,y_test.shape)
			dtloadend = time.time()
			print('\n Test Data Load Time : ',dtloadend-dtloadstart, ' seconds')
			if m_test_typ == 'I':
				model_type = self.best_model
			else:
				model_type = self.model_name			
			model.load_weights(self.MODELS_PATH + self.model_name + model_type + self.ind + '_' + self.run_count +'.h5')


		tststart = time.time()
		self.testmodel(model,x_test,y_test,x_val,y_val,xtscas,ytscase)
		tstend = time.time()
		print('\nTesting Time : ',(tstend-tststart), ' seconds')


	def testmodel(self,model,x_test,y_test,x_val,y_val,xtscas,ytscase):

		print('\n\n',15*'* ','Eval',15*' *')

		epred = model.predict(x_val)

		y_epred = np.round(epred)
		# y_epred = to_categorical(epred)

		precision = precision_score(y_val, y_epred,average='macro')
		recall = recall_score(y_val, y_epred,average='macro')
		f1score = f1_score(y_val, y_epred,average='macro')
		cf = confusion_matrix(np.argmax(y_val,axis=-1), np.argmax(y_epred,axis=-1))

		print("Eval precision is : " + str(precision))
		print("Eval recall is : " + str(recall))
		print("Eval F1 is : " + str(f1score))
		print("Eval Accuracy is : " + str(accuracy_score(y_val, y_epred)))
		print('\nEval Confusion Matrix : \n',cf)

		print('\n\n',15*'* ','Test',15*' *')
		ttstst = time.time()
		pred = model.predict(x_test)
		# print(pred.shape,pred,'\n')

		y_pred = np.round(pred)
		# y_pred = to_categorical(pred)
		ttsted = time.time()
		print('\nTest Time : ',(ttsted-ttstst))

		# for l,p,o in zip(y_test,pred,y_pred):
		#   print('\n',l,'\t',p,'\t',o)

		precision = precision_score(y_test, y_pred,average='macro')
		recall = recall_score(y_test, y_pred,average='macro')
		f1score = f1_score(y_test, y_pred,average='macro')
		cf = confusion_matrix(np.argmax(y_test,axis=-1), np.argmax(y_pred,axis=-1))


		print("Test precision is : " + str(precision))
		print("Test recall is : " + str(recall))
		print("Test F1 is : " + str(f1score))
		print("Test Accuracy is : " + str(accuracy_score(y_test, y_pred)))
		print('\nTest Confusion Matrix : \n',cf)


		print('\n\nCase Wise Test Result')

		for xtyp,ytyp in zip(xtscas,ytscase):
			ipred = 0
			hpred = 0
			npred = 0
			hipred = 0
			oocpred =0      
			k=np.argmax(ytyp)
			if k==0:
				print('\nIsc: ',len(xtyp))
			elif k==1:
				print('\nHemo: ',len(xtyp))
			elif k==2:
				print('\nNormal: ',len(xtyp))
			# print('\nTotal Cases: ',len(xtyp))
			
			for i,case in enumerate(xtyp):
				spred = []
				result = []
				# print(len(case))
				for slc in case:
					# print(slc.shape)
					slc = np.expand_dims(slc, axis=0)
					# print(slc.shape)
					# slc = np.moveaxis(slc,1,2)
					# print(slc.shape)
					# slc = np.moveaxis(slc,2,3)
					# print(slc.shape)
					# res = model.predict(slc)
					res = model.predict(slc)
					# print(res,np.round(res),np.argmax(np.round(res)))
					spred.append(np.argmax(np.round(res)))
				# print(res,'\n',spred)
				spred = np.array(spred)
				if np.all(spred == 2, axis=-1):
					# print('Case ',i,' :N') 
					npred+=1
				elif np.any(spred == 0, axis=-1) and np.any(spred == 1, axis=-1):
					# print('Case ',i,' :Both I &H') 
					ih = np.bincount(spred)
					if ih[0]>ih[1]:
						ipred+=1
					elif ih[0]<ih[1]:
						hpred+=1
					else:
						hipred+=1
				elif np.any(spred == 0, axis=-1):
					# print('Case ',i,' :I') 
					ipred+=1
				elif np.any(spred == 1, axis=-1):
					# print('Case ',i,' :H') 
					hpred+=1
				else:
					oocpred+=1
					print('Case ',i,' :Out Of Class') 
			print('\nH-Count',hpred)
			print('\nI-Count',ipred)
			print('\nN-Count',npred)
			print('\nHI-Count',hipred)
			print('\nOOC-Count',oocpred)



		print('\nClassification Report')
		cname = ['Ischemic','Hemorrhagic','Normal']
		print(classification_report(np.argmax(y_test,axis=-1), np.argmax(y_pred,axis=-1),target_names=cname))




if __name__ == '__main__':
	mdlstart = time.time()
	# parser = argparse.ArgumentParser()
	# parser.add_argument("g",help="Enter the GPU ID")
	# parser.add_argument("s",help="Enter the slice# per sample")
	# parser.add_argument("d",help="Enter the layer depth of architecture")
	# parser.add_argument("p",help="Enter the Data Set Type (Custom/BW)")
	# parser.add_argument("t",help="Enter Y to enable train else N")
	# parser.add_argument("r",help="Enter Experiment Run Count")
	

	# args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]='6'
	inputensor = Input((256,256, 3, 1), name='img')
	n_classes = 6
	resv2obj = ResNetV2(inputensor,n_classes,50) #'101',16)#
	resv2model = resv2obj.createmodelv2()
	print(resv2model.summary())
	# mdlend = time.time()
	# print('\nModel Load Time : ',mdlend-mdlstart, ' seconds')
	# trainobj = trainortest('6',True, 'RSNA', '3', '1','50')
	# trainobj.startTrainingandTesting(resv2model)

	# if args.t == 'Y':
	# 	isTrain = True
	# 	trainobj = trainortest(args.g,isTrain, args.p, args.s, args.r,args.d)
	# 	trainobj.startTrainingandTesting(resv2model)
	# else:
	# 	isTrain = False
	# 	trainobj = trainortest(args.g,isTrain, args.p, args.s, args.r,args.d)
	# 	trainobj.startTrainingandTesting(resv2model,m_test_typ='I')

	

