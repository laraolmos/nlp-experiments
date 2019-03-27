# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import re
import codecs
from bs4 import BeautifulSoup


# Extract text from HTML file

class TextExtractor():

	# clean HTML line
	def clean_html(self, raw_html):
		soup = BeautifulSoup(raw_html, 'lxml')
		cleantext = soup.get_text()
		cleanr = re.compile('<.*?>')
		cleantext = re.sub(cleanr, ' ', cleantext)
		return cleantext

	# all content HTML file in one string
	def extract_html_file_content(self, file_route):
		content_lines = []
		with codecs.open(file_route, 'r', encoding='utf-8')  as content_file:
			raw_html = content_file.read()
			extracted_text = self.clean_html(raw_html)
			content_lines.append(extracted_text)
		return ' '.join((content_lines))


