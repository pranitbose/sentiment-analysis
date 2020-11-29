# -*- coding: utf-8 -*-
# @Author: pranit
# @Date:   2018-05-16 10:28:28
# @Last Modified by:   pranit
# @Last Modified time: 2018-05-16 21:04:34

from sentiment_analyzer import SentimentAnalyzer

import pickle

class Utility:

	def __init__(self):
		self.sentiment = SentimentAnalyzer()
		self.clf = self.sentiment.clf

	def classifiersVsFeatures(self):
		with open('pickled/features_train.pickle', 'rb') as features_train:
			X_train = pickle.load(features_train)
		with open('pickled/features_test.pickle', 'rb') as features_test:
			X_test = pickle.load(features_test)
		with open('pickled/labels_train.pickle', 'rb') as labels_train:
			y_train = pickle.load(labels_train)
		with open('pickled/labels_test.pickle', 'rb') as labels_test:
			y_test = pickle.load(labels_test)

		num_features = [10000, 50000, 100000, 500000, 1000000]
		
		acc = []
		for i in range(0, len(self.clf)):
			acc.append([])

		for k in num_features:
			_, model = self.sentiment.trainData(X_train, y_train, self.clf, k)
			prediction = self.sentiment.predictData(X_test, model)
			clf_metrics = self.sentiment.evaluate(y_test, prediction)

			for j in range(0, len(self.clf)):
				print(clf_metrics[0][j])
				acc[j].append(clf_metrics[0][j]) # Append the accuracy of the classifier for each k

		data = []
		for i in range (0, len(self.clf)):
			data.append({'x': num_features, 'y': acc[i]})

		return data

	def showTopFeatures(self, pipeline, n = 20):
		vectorizer = pipeline.named_steps['vect']
		clf = pipeline.named_steps['clf']
		feature_names = vectorizer.get_feature_names()

		coefs = sorted(zip(clf.coef_[0], feature_names), reverse = True)
		topn = zip(coefs[:n], coefs[: -(n+1): -1])
		
		top_features = []
		for (coef_p, feature_p), (coef_n, feature_n) in topn:
			top_features.append('{:0.4f}{: >25}    {:0.4f}{: >25}'.format(coef_p, feature_p, coef_n, feature_n))

		return '\n'.join(top_features)
