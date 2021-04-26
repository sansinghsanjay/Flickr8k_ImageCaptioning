'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
Image Captioning by Attention Mechanism
Implementation of:
https://github.com/SubhamIO/Image-Captioning-using-Attention-Mechanism-Local-Attention-and-Global-Attention-/blob/master/Image_Captioning_using_Attention_Mechanism.ipynb
'''

# packages
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
#from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd 
from PIL import Image
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from os import listdir

# constants
embedding_dim = 256
units = 512
vocab_size = 8329 #len(tokenizer.word_index) + 1 #8329
num_steps = 500 #len(img_name_train) // BATCH_SIZE  #500
EPOCHS = 20
# Shape from last layer of VGG-16 :(7,7,512)
# So, say there are 49 pixel locations now and each pixel is 512 dimensional
features_shape = 512
attention_features_shape = 49 

# https://www.tensorflow.org/tutorials/text/image_captioning
class VGG16_Encoder(tf.keras.Model):
	# This encoder passes the features through a Fully connected layer
	def __init__(self, embedding_dim):
		super(VGG16_Encoder, self).__init__()
		# shape after fc == (batch_size, 49, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)
		self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
	def call(self, x):
		#x= self.dropout(x)
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x

def rnn_type(units):
	# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
	# the code automatically does that.
	if tf.test.is_gpu_available():
		return tf.compat.v1.keras.layers.CuDNNLSTM(units, 
										return_sequences=True, 
										return_state=True, 
										recurrent_initializer='glorot_uniform')
	else:
		return tf.keras.layers.GRU(units, 
									return_sequences=True, 
									return_state=True, 
									recurrent_activation='sigmoid', 
									recurrent_initializer='glorot_uniform')

'''The encoder output(i.e. 'features'), hidden state(initialized to 0)(i.e. 'hidden') and 
the decoder input (which is the start token)(i.e. 'x') is passed to the decoder.'''
class Rnn_Local_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(Rnn_Local_Decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
									return_sequences=True,
									return_state=True,
									recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
		self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		# Implementing Attention Mechanism 
		self.Uattn = tf.keras.layers.Dense(units)
		self.Wattn = tf.keras.layers.Dense(units)
		self.Vattn = tf.keras.layers.Dense(1)

	def call(self, x, features, hidden):
		# features shape ==> (64,49,256) ==> Output from ENCODER
		# hidden shape == (batch_size, hidden_size) ==>(64,512)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)
		# score shape == (64, 49, 1)
		# Attention Function
		'''e(ij) = f(s(t-1),h(j))'''
		''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
		score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
		# self.Uattn(features) : (64,49,512)
		# self.Wattn(hidden_with_time_axis) : (64,1,512)
		# tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
		# self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
		# you get 1 at the last axis because you are applying score to self.Vattn
		# Then find Probability using Softmax
		'''attention_weights(alpha(ij)) = softmax(e(ij))'''
		attention_weights = tf.nn.softmax(score, axis=1)
		# attention_weights shape == (64, 49, 1)
		# Give weights to the different pixels in the image
		''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) ''' 
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)
		# Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
		# context_vector shape after sum == (64, 256)
		# x shape after passing through embedding == (64, 1, 256)
		x = self.embedding(x)
		# x shape after concatenation == (64, 1,  512)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)
		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)
		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))
		# Adding Dropout and BatchNorm Layers
		x= self.dropout(x)
		x= self.batchnormalization(x)
		# output shape == (64 * 512)
		x = self.fc2(x)
		# shape : (64 * 8329(vocab))
		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))

''' To choose score type: 
	Enter 'dot' for dot score
	Enter 'general' for general score
	Enter 'concat' for concat score
'''
def score_choose():
	scoring_type= input('Enter the scoring method: ')
	return scoring_type

scoring_type=score_choose()

class Rnn_Global_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size,scoring_type):
		super(Rnn_Global_Decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
								return_sequences=True,
								return_state=True,
								recurrent_initializer='glorot_uniform')
		self.wc = tf.keras.layers.Dense(units, activation='tanh')
		self.ws = tf.keras.layers.Dense(vocab_size)
		#For Attention
		self.wa = tf.keras.layers.Dense(units)
		self.wb = tf.keras.layers.Dense(units)
		#For Score 3 i.e. Concat score
		self.Vattn = tf.keras.layers.Dense(1)
		self.wd = tf.keras.layers.Dense(units, activation='tanh')
		self.scoring_type = scoring_type

	def call(self, sequence, features,hidden):
		# features : (64,49,256)
		# hidden : (64,512)
		embed = self.embedding(sequence)
		# embed ==> (64,1,256) ==> decoder_input after embedding (embedding dim=256)
		output, state = self.gru(embed)       
		#output :(64,1,512)
		score=0
		#Dot Score as per paper(Dot score : h_t (dot) h_s') (NB:just need to tweak gru units to 256)
		'''----------------------------------------------------------'''
		if(self.scoring_type=='dot'):
			xt=output #(64,1,512)
			xs=features #(256,49,64)  
			score = tf.matmul(xt, xs, transpose_b=True) 
			#score : (64,1,49)
		'''----------------------------------------------------------'''
		'''----------------------------------------------------------'''
		# General Score as per Paper ( General score: h_t (dot) Wa (dot) h_s')
		'''----------------------------------------------------------'''
		if(self.scoring_type=='general'):
			score = tf.matmul(output, self.wa(features), transpose_b=True)
			# score :(64,1,49)
		'''----------------------------------------------------------'''
		'''----------------------------------------------------------'''
		# Concat score as per paper (score: VT*tanh(W[ht;hs']))    
		'''----------------------------------------------------------'''
		#https://www.tensorflow.org/api_docs/python/tf/tile
		if(self.scoring_type=='concat'):
			tiled_features = tf.tile(features, [1,1,2]) #(64,49,512)
			tiled_output = tf.tile(output, [1,49,1]) #(64,49,512)
			concating_ht_hs = tf.concat([tiled_features,tiled_output],2) ##(64,49,1024)
			tanh_activated = self.wd(concating_ht_hs)
			score =self.Vattn(tanh_activated)
			#score :(64,49,1), but we want (64,1,49)
			score= tf.squeeze(score, 2)
			#score :(64,49)
			score = tf.expand_dims(score, 1)
			#score :(64,1,49)
		'''----------------------------------------------------------'''
		'''----------------------------------------------------------'''
		# alignment vector a_t
		alignment = tf.nn.softmax(score, axis=2)
		# alignment :(64,1,49)
		# context vector c_t is the average sum of encoder output
		context = tf.matmul(alignment, features)
		# context : (64,1,256)
		# Combine the context vector and the LSTM output
		output = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)
		# output: concat[(64,1,256):(64,1,512)] = (64,768)
		output = self.wc(output)
		# output :(64,512)
		# Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
		logits = self.ws(output)
		# logits/predictions: (64,8239) i.e. (batch_size,vocab_size))
		return logits, state, alignment

		def reset_state(self, batch_size):
			return tf.zeros((batch_size, self.units))

'''
Type: 'global' for Luong's Attention Mechanism(Global Attention)
Type: 'local' for Bahdanau's Attention Mechanism(Local Attention)
'''
attention_choice = input("Choose the type of Attention Mechanism you want to apply :")

if(attention_choice=='local'):
	encoder = VGG16_Encoder(embedding_dim)
	decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)
else:
	encoder = VGG16_Encoder(embedding_dim)
	decoder = Rnn_Global_Decoder(embedding_dim, units, vocab_size,scoring_type)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tf.reduce_mean(loss_)

start_epoch = 0

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

loss_plot = []

@tf.function
def train_step(img_tensor, target):
	loss = 0
	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])
	dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
	with tf.GradientTape() as tape:
		features = encoder(img_tensor)
		for i in range(1, target.shape[1]):
			# passing the features through the decoder
			#dec_input = tf.expand_dims(target[:, i], 1)
			predictions, hidden, _ = decoder(dec_input, features, hidden)
			loss += loss_function(target[:, i], predictions)
			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)
	total_loss = (loss / int(target.shape[1]))
	trainable_variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))
	#train_loss(loss)
	#train_accuracy(target, predictions)
	return loss, total_loss
