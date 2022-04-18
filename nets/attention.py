import math

import tensorflow as tf
from keras import backend as K
from keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, BatchNormalization,
                          Reshape, multiply)


def se_block(input_feature, ratio=16, name=""):
	channel = input_feature._keras_shape[-1]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)

	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_one_"+str(name))(se_feature)
					   
	se_feature = Dense(channel,
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_two_"+str(name))(se_feature)
	se_feature = Activation('sigmoid')(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def channel_attention(input_feature, ratio=8, name=""):
	
	channel = input_feature._keras_shape[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_one_"+str(name))
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_two_"+str(name))
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	max_pool = GlobalMaxPooling2D()(input_feature)

	avg_pool = Reshape((1,1,channel))(avg_pool)
	max_pool = Reshape((1,1,channel))(max_pool)

	avg_pool = shared_layer_one(avg_pool)
	max_pool = shared_layer_one(max_pool)

	avg_pool = shared_layer_two(avg_pool)
	max_pool = shared_layer_two(max_pool)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, name=""):
	kernel_size = 7

	cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	concat = Concatenate(axis=3)([avg_pool, max_pool])

	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					name = "spatial_attention_"+str(name))(concat)	
	cbam_feature = Activation('sigmoid')(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature

def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = input_feature._keras_shape[-1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	
	avg_pool = GlobalAveragePooling2D()(input_feature)
	
	x = Reshape((-1,1))(avg_pool)
	x = Conv1D(1, kernel_size=kernel_size, padding="same", name = "eca_layer_"+str(name), use_bias=False,)(x)
	x = Activation('sigmoid')(x)
	x = Reshape((1, 1, -1))(x)

	output = multiply([input_feature,x])
	return output

def ca_block(input_feature, ratio=16, name=""):
	channel = input_feature._keras_shape[-1]
	h		= input_feature._keras_shape[1]
	w		= input_feature._keras_shape[2]
 
	x_h = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
	x_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_h)
	x_w = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)
	
	x_cat_conv_relu = Concatenate(axis=2)([x_w, x_h])
	x_cat_conv_relu = Conv2D(channel // ratio, kernel_size=1, strides=1, use_bias=False, name = "ca_block_conv1_"+str(name))(x_cat_conv_relu)
	x_cat_conv_relu = BatchNormalization(name = "ca_block_bn_"+str(name))(x_cat_conv_relu)
	x_cat_conv_relu = Activation('relu')(x_cat_conv_relu)
 
	x_cat_conv_split_h, x_cat_conv_split_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(x_cat_conv_relu)
	x_cat_conv_split_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_cat_conv_split_h)
	x_cat_conv_split_h = Conv2D(channel, kernel_size=1, strides=1, use_bias=False, name = "ca_block_conv2_"+str(name))(x_cat_conv_split_h)
	x_cat_conv_split_h = Activation('sigmoid')(x_cat_conv_split_h)
 
	x_cat_conv_split_w = Conv2D(channel, kernel_size=1, strides=1, use_bias=False, name = "ca_block_conv3_"+str(name))(x_cat_conv_split_w)
	x_cat_conv_split_w = Activation('sigmoid')(x_cat_conv_split_w)
 
	output = multiply([input_feature, x_cat_conv_split_h])
	output = multiply([output, x_cat_conv_split_w])
	return output
