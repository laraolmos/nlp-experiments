# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import codecs
import pickle
from pickle import Unpickler

from settings import OUTPUT_DIR


def export_to_txt(file_name, text):
	with codecs.open(OUTPUT_DIR + file_name, 'w', encoding='utf-8') as f:
		f.write(text)

def add_to_txt(file_name, text):
	with codecs.open(file_name, 'a', encoding='utf-8') as f:
		f.write(text + '\n')

def export_dictionary_to_txt(filename, dictionary):
	str_list = [str(key) + ' -> ' + str(dictionary[key]) for key in sorted(dictionary.keys())]
	export_to_txt(filename, '\n'.join(str_list))

def load_from_txt(file_route):
	with codecs.open(file_route, 'r', encoding='utf-8') as f:
		text = f.read()
	return text

def export_pickle(file_name, data):
	with open(file_name, 'wb') as pickle_file:
		pickle.dump(data, pickle_file, protocol=2)

def load_pickle(input_file):
	pickle_file = open(input_file, 'rb')
	pickler_obj = Unpickler(pickle_file)
	result = pickler_obj.load()
	pickle_file.close()
	return result