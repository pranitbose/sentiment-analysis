# -*- coding: utf-8 -*-
# @Author: pranit
# @Date:   2018-05-14 10:31:38
# @Last Modified by:   pranit
# @Last Modified time: 2018-05-15 08:06:52

import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import sent_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag

class NltkPreprocessor:

	def __init__(self, stopwords = None, punct = None, lower = True, strip = True):
		self.lower = lower
		self.strip = strip
		self.stopwords = stopwords or set(sw.words('english'))
		self.punct = punct or set(string.punctuation)
		self.lemmatizer = WordNetLemmatizer()

	def tokenize(self, document):
		tokenized_doc = []

		for sent in sent_tokenize(document):
			for token, tag in pos_tag(wordpunct_tokenize(sent)):
				token = token.lower() if self.lower else token
				token = token.strip() if self.strip else token
				token = token.strip('_0123456789') if self.strip else token

				if token in self.stopwords:
					continue

				if all(char in self.punct for char in token):
					continue

				lemma = self.lemmatize(token, tag)
				tokenized_doc.append(lemma)

		return ' '.join(tokenized_doc)

	def lemmatize(self, token, tag):
		tag = {
			'N': wn.NOUN,
			'V': wn.VERB,
			'R': wn.ADV,
			'J': wn.ADJ
		}.get(tag[0], wn.NOUN)

		return self.lemmatizer.lemmatize(token, tag)
