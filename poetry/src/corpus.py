# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'

import os

from settings import LMDL_CORPUS_POETRY
from extractor import *
from preprocessor import *

from export import export_to_txt


class LMDL_Corpus():

	def __init__(self, input_path=LMDL_CORPUS_POETRY, filter_opt=True):
		self.input_path = input_path
		self.text_extractor = TextExtractor()
		self.text_tokenizer = TextTokenizer()
		self.text_normalizer = TextNormalizer()
		self.corpus, self.vocabulary, self.corpus_processed = self._build(filter_opt=filter_opt)

	def _build(self, filter_opt=True):
		dir_list_files = sorted(os.listdir(self.input_path))
		original_text = {}
		processed_text = {}
		vocabulary = []
		for file_name in dir_list_files:
			original_text[file_name] = self.text_extractor.extract_html_file_content(self.input_path + file_name)
			document_terms = self.text_normalizer.normalize_token_list(self.text_tokenizer.words(original_text[file_name]))
			if filter_opt:
				document_terms = self.text_normalizer.filter_token_list(document_terms)
			processed_text[file_name] = document_terms
			for token in processed_text[file_name]:
				if token not in vocabulary:
					vocabulary.append(token)
		return original_text, vocabulary, processed_text

	def sentences(self, doc_name):
		return self.text_tokenizer.sentences(self.corpus[doc_name])

	def words(self, doc_name):
		return self.text_tokenizer.words(self.corpus[doc_name])

	def document_terms(self, doc_name):
		return self.filter_token_list(self.text_normalizer.normalize_token_list(self.words(doc_name)))

	def get_filenames(self):
		return sorted(list(self.corpus.keys()))

	def get_original_texts(self):
		return list(self.corpus.values())

	def get_processed_documents(self):
		return self.corpus_processed.values()

	def get_vocabulary(self):
		return self.vocabulary

	def vocabulary_size(self):
		return len(self.vocabulary)

	def number_of_documents(self):
		return len(self.corpus.keys())


if __name__ == '__main__':
	corpus = LMDL_Corpus()
	print(corpus.vocabulary_size())
