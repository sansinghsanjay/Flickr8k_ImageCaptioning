'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To train a neural network for Image Captioning - on Google Colab
This script file is copy of script_training.py, only paths have changed in this file to make it to work on Google Colab
'''

# packages
import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# constants
EPOCHS = 20
NUMBER_PICS_PER_BATCH = 3
EMBEDDING_DIM = 200

# function to update status
def percentage_progress(completed, total):
	perc_progress = (completed / total) * 100
	perc_progress = round(perc_progress, 2)
	return perc_progress

# data generator function
'''
To run parts of this function in ipython without calling function
image_name_captions = train_image_caption
image_features = train_image_feature
num_pics_per_batch = NUMBER_PICS_PER_BATCH
'''
def data_generator(image_name_captions, image_features, wordtoix, max_caption_length, vocab_size, num_pics_per_batch):
	X1, X2, y = list(), list(), list()
	n = 0
	while(True):
		for i in range(image_name_captions.shape[0]):
			n += 1
			image_name = image_name_captions.iloc[i]['image']
			image_feature = image_features[image_name]
			captions = image_name_captions.iloc[i]['caption']
			captions_list = captions.split("#")
			for caption in captions_list:
				seq = [wordtoix[word] for word in caption.split(' ') if word in wordtoix]
				for j in range(len(seq)):
					in_seq, out_seq = seq[:j], seq[j]
					in_seq = pad_sequences([in_seq], maxlen=max_caption_length, padding='post')[0]
					out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
					X1.append(image_feature)
					X2.append(in_seq)
					y.append(out_seq)
			if(n == num_pics_per_batch):
				yield [array(X1), array(X2)], array(y)
				X1, X2, y = list(), list(), list()
				n = 0

# paths
train_image_caption_path = "../processed_data/train_image_caption_processed.csv"
train_image_bottleneck_feature_path = "../processed_data/train_imagename_bottleneck_feat.csv"
vocabulary_path = "../processed_data/vocabulary.txt"
glove_model_path = "../pre-trained_models/glove.6B.200d.txt"
max_caption_length_path = "../processed_data/max_caption_length.txt"
target_path = "../output/trained_models/"

# reading vocabulary
vocabulary = list()
f_ptr = open(vocabulary_path, 'r')
lines = f_ptr.readlines()
for line in lines:
	vocabulary.append(line.strip())
print("Completed reading vocabulary file")

# creating word-to-index and index-to word dictionary
wordtoix = dict()
ixtoword = dict()
ix = 1
for i in range(len(vocabulary)):
	wordtoix[vocabulary[i]] = ix
	ixtoword[ix] = vocabulary[i]
	ix += 1
print("Created word-to-index and index-to-word dictionary")

# getting vocabulary size
vocab_size = len(wordtoix) + 1 # 1 is added for '0'
print("Vocabulary Size: ", vocab_size)

# loading GloVe model
glove_model = dict()
glove_data = open(glove_model_path, encoding='utf-8')
for line in glove_data:
	values = line.split()
	word = values[0]
	feat = values[1:]
	glove_model[word] = feat
print("Loaded GloVe model")

# creating embedding matrix, i.e., glove feature for each word in vocabulary
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM)) 
for i in range(len(vocabulary)):
	embedding_vec = glove_model.get(vocabulary[i])
	if(embedding_vec is not None):
		embedding_matrix[i] = embedding_vec
print("Successfully created Embedding Matrix, i.e., GloVe bottleneck features for each word of vocabulary")

# reading training file (image name and processed captions)
train_image_caption = pd.read_csv(train_image_caption_path)
print("Successfully read training - image name and processed caption file")

# reading training - image name and bottleneck feature file
train_image_feature = pd.read_csv(train_image_bottleneck_feature_path)
print("Sucessfully read training - image name and their InceptionV3 bottleneck feature file")

# converting train_image_feature dataframe to dictionary
print("Converting training - image name and feature from dataframe to dicitonary for quick search. Wait...")
train_image_feature = train_image_feature.set_index('image').T.to_dict('list')
print("Successfully converted training - image name and feature from dataframe to dictionary")

# steps required for training
steps = (train_image_caption.shape[0]) // NUMBER_PICS_PER_BATCH

# extract maximum length of caption - saved by script_preprocessing.py
f_ptr = open(max_caption_length_path, 'r')
line = f_ptr.readlines()
max_caption_length = int(line[0].split(":")[1].strip())

# passing data to generator function


# defining model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_caption_length,))
se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# print model summary
model.summary()

# use embedding matrix as weights in layer 2
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train network
for i in range(EPOCHS):
	generator = data_generator(train_image_caption, train_image_feature, wordtoix, max_caption_length, vocab_size, NUMBER_PICS_PER_BATCH)
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	model.save(target_path + 'model_' + str(i) + '.h5')

# tuning learning rate
model.optimizer.lr = 0.0001
NEW_EPOCHS = 10
NEW_NUMBER_PICS_PER_BATCH = 6
steps = (train_image_caption.shape[0]) // NEW_NUMBER_PICS_PER_BATCH

# train network
for i in range(NEW_EPOCHS):
	generator = data_generator(train_image_caption, train_image_feature, wordtoix, max_caption_length, vocab_size, NEW_NUMBER_PICS_PER_BATCH)
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	model.save(target_path + 'model_' + str(EPOCHS + i) + '.h5')
