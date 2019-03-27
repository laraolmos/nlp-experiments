# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import load_model

from settings import *
from corpus import *
from export import *



class PoetryModel():

	def __init__(self, train=True):
		self.seq_length = 75
		if train:
			self.raw_text = load_from_txt(LMDL_POETRY_FILE)

	def _generate_dataset(self):
		corpus = LMDL_Corpus(input_path=LMDL_CORPUS_POETRY, filter_opt=False)
		raw_text = '\n'.join([' '.join(text_token) for text_token in corpus.get_processed_documents()])
		export_to_txt('poetry-dataset.txt', raw_text)

	def _compile_model(self):
		pass

	def _train_model(self):
		pass

	def generate(self):
		pass


class Model():

	def model_two_lstm(self, X, y, size=256):
		model = Sequential()
		model.add(LSTM(size, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
		model.add(Dropout(0.2))
		model.add(LSTM(size))
		model.add(Dropout(0.2))
		model.add(Dense(y.shape[1], activation='softmax'))
		return model

	def model_two_bilstm(self, X, y, size=256):
		model = Sequential()
		model.add(Bidirectional(LSTM(size, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
		model.add(Dropout(0.2))
		model.add(Bidirectional(LSTM(size)))
		model.add(Dropout(0.2))
		model.add(Dense(y.shape[1], activation='softmax'))
		return model

	def model_multiple_bilstm(self, X, y, size=128, num_cells=6):
		model = Sequential()
		model.add(Bidirectional(LSTM(size, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
		model.add(Dropout(0.2))
		for i in range(1, num_cells-1):
			model.add(Bidirectional(LSTM(size, return_sequences=True)))
			model.add(Dropout(0.2))
		model.add(Bidirectional(LSTM(size)))
		model.add(Dropout(0.2))
		model.add(Dense(y.shape[1], activation='softmax'))
		return model


class LSTMCharacterModel(PoetryModel):

	def __init__(self, train=True, weights_filename=None, model_save=POETRY_MODEL, mappings=POETRY_MAPPINGS):
		PoetryModel.__init__(self, train=train)
		self.modelClass = Model()
		if train:
			self.n_chars, self.n_vocab, self.char_to_int, self.int_to_char = self._create_mapping()
			n_patterns, self.dataX, dataY = self._encode_input(self.raw_text, self.n_chars, self.char_to_int, self.int_to_char)
			self.model, X, y = self._compile_model(self.dataX, dataY, n_patterns, self.n_vocab, filename=weights_filename)
			self.model = self._train_model(self.model, X, y)
			self.model.save(model_save)
			export_pickle(mappings, (self.char_to_int, self.int_to_char, self.dataX, self.n_vocab, self.n_chars))
		else:
			data_compact = load_pickle(mappings)
			self.char_to_int = data_compact[0]
			self.int_to_char = data_compact[1] 
			self.dataX = data_compact[2]
			self.n_vocab = data_compact[3]
			self.n_chars = data_compact[4]
			self.model = load_model(model_save)

	def _create_mapping(self):
		# create mapping of unique chars to integers, and a reverse mapping
		chars = sorted(list(set(self.raw_text)))
		char_to_int = dict((c, i) for i, c in enumerate(chars))
		int_to_char = dict((i, c) for i, c in enumerate(chars))
		# summarize the loaded data
		n_chars = len(self.raw_text)
		n_vocab = len(chars)
		return n_chars, n_vocab, char_to_int, int_to_char
	
	def _encode_input(self, input_text, n_chars, char_to_int, int_to_char):
		# prepare the dataset of input to output pairs encoded as integers
		dataX = []
		dataY = []
		for i in range(0, n_chars - self.seq_length, 1):
			seq_in = input_text[i:i + self.seq_length]
			seq_out = input_text[i + self.seq_length]
			dataX.append([char_to_int[char] for char in seq_in])
			dataY.append(char_to_int[seq_out])
		n_patterns = len(dataX)
		return n_patterns, dataX, dataY

	def _compile_model(self, dataX, dataY, n_patterns, n_vocab, filename=None):
		# reshape X to be [samples, time steps, features]
		X = numpy.reshape(dataX, (n_patterns, self.seq_length, 1))
		# normalize
		X = X / float(n_vocab)
		# one hot encode the output variable
		y = np_utils.to_categorical(dataY)
		# define the LSTM model
		model = self.modelClass.model_multiple_bilstm(X, y)
		if filename:
			model.load_weights(filename)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		model.summary()
		return model, X, y

	def _train_model(self, model, X, y, epochs=NUM_EPOCHS, batch_size=128, filename=POETRY_WEIGHTS):
		# define the checkpoint
		checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		# fit the model
		model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
		return model

	def _pick_random_seed(self):
		start = numpy.random.randint(0, len(self.dataX)-1)
		pattern = self.dataX[start]
		pattern_output = ''.join([self.int_to_char[value] for value in pattern])
		print("Seed:")
		print("\"" + pattern_output + "\"")
		add_to_txt(LMDL_POETRY_DIR + 'response-poetry.txt', '\nseed: ' + pattern_output)
		return pattern

	def _mapping(self, text):
		return [self.char_to_int[input_char] for input_char in text]

	def generate(self, generation_range=100, pattern=None, generation_file=POETRY_GENERATED):
		response_result = ''
		if not pattern:
			pattern = self._pick_random_seed()
		else:
			n_chars = len(pattern)
			add_to_txt(LMDL_POETRY_DIR + 'response-poetry.txt', '\nseed: ' + pattern)
			pattern = self._mapping(pattern)
		for i in range(generation_range):
			x = numpy.reshape(pattern, (1, len(pattern), 1))
			x = x / float(self.n_vocab)
			prediction = self.model.predict(x, verbose=0)
			index = numpy.argmax(prediction)
			result = self.int_to_char[index]
			response_result += result
			pattern.append(index)
			pattern = pattern[1:len(pattern)]
		add_to_txt(generation_file, 'maq: ' + response_result)
		print(response_result)



if __name__ == '__main__':

	#poetry = LSTMCharacterModel()
	#poetry = LSTMCharacterModel(weights_filename=POETRY_WEIGHTS)
	poetry = LSTMCharacterModel(train=False)
	poetry.generate()
	#poetry.generate(pattern='y como en tu caminar cansado miras al frente con paso firme sin tu descanso')
	#poetry.generate(pattern='más hacía tiempo que el tiempo nos había separado el tiempo fue pasando sin')
