# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import re
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


from settings import SPANISH_STOPWORDS


# Segment text in phrases or words

class TextTokenizer():

	def sentences(self, input_text):
		spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
		return spanish_tokenizer.tokenize(input_text)

	def words(self, input_text):
		return word_tokenize(input_text)


# Process after tokenization

class TextNormalizer():

	def __init__(self):
		self.stemmer = SnowballStemmer('spanish')

	def _remove_puntuation(self, word):
		regular_expr = re.compile('\r|\n|\t|\(|\)|\[|\]|:|\.|\,|\;|"|”|…|»|“|/|\'|\?|\¿|\!|\¡|`|\%|\.\.\.|-|—|=|–|―')
		word_processed = re.sub(regular_expr, '', word)
		return word_processed

	def _remove_numbers(self, word):
		num_expr = re.compile('[0-9]+|[0-9]*[,.][0-9]+')
		word_processed = re.sub(num_expr, '', word)
		return word_processed

	def _remove_stopwords(self, sentence):
		return [token for token in sentence if token not in SPANISH_STOPWORDS]

	def _filter_words(self, sentence):
		return [token for token in sentence if token not in [' ', ''] and len(token) > 1 and len(token) < 10 ]

	def _stemming(self, token):
		return self.stemmer.stem(token)

	def _replace_accents(self, token):
		pass

	def _lemmatize(self, token):
		pass

	def normalize_token_list(self, token_list):
		processed_token_list = []
		for word in token_list:
			word_processed = word.lower().strip()
			word_processed = self._remove_puntuation(word_processed)
			word_processed = self._remove_numbers(word_processed)
			#word_processed = self._stemming(word_processed)
			#word_processed = self._replace_accents(word_processed)			
			#word_processed = self._lemmatize(word_processed)
			processed_token_list.append(word_processed)
		return self._filter_words(processed_token_list)

	def filter_token_list(self, processed_token_list):
		filtered_token_list = self._remove_stopwords(filtered_token_list)
		return filtered_token_list

	def normalize_corpus(self, corpus):
		return [self.normalize_token_list(token_list) for token_list in corpus]
