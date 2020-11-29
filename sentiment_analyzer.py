# -*- coding: utf-8 -*-
# @Author: pranit
# @Date:   2018-04-20 09:59:48
# @Last Modified by:   pranit
# @Last Modified time: 2018-05-17 02:16:39

from time import time
import ast
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

from preprocessor import NltkPreprocessor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

class SentimentAnalyzer:

	def __init__(self):
		self.clf = [
			('MNB', MultinomialNB(alpha = 1.0, fit_prior = False)),
			('LR', LogisticRegression(C = 5.0, penalty = 'l2', solver = 'liblinear', max_iter = 100, dual = True)),
			('SVM', LinearSVC(C = 0.55, penalty = 'l2', max_iter = 1000, dual = True)),
			('RF', RandomForestClassifier(n_jobs = -1, n_estimators = 100, min_samples_split = 40, max_depth = 90, min_samples_leaf = 3))
		]
		self.clf_names = ['Multinomial NB', 'Logistic Regression', 'Linear SVC', 'Random Forest']

	def getInitialData(self, data_file, do_pickle):
		print('Fetching initial data...')
		t = time()

		i = 0
		df = {}
		with open(data_file, 'r') as file_handler:
			for review in file_handler.readlines():
				df[i] = ast.literal_eval(review)
				i += 1

		reviews_df = pd.DataFrame.from_dict(df, orient = 'index')
		if do_pickle:
			reviews_df.to_pickle('pickled/product_reviews.pickle')

		print('Fetching data completed!')
		print('Fetching time: ', round(time()-t, 3), 's\n')

	def preprocessData(self, reviews_df, do_pickle):
		print('Preprocessing data...')
		t = time()

		reviews_df.drop(columns = ['reviewSummary'], inplace = True)
		reviews_df['reviewRating'] = reviews_df.reviewRating.astype('int')

		reviews_df = reviews_df[reviews_df.reviewRating != 3] # Ignoring 3-star reviews -> neutral
		reviews_df = reviews_df.assign(sentiment = np.where(reviews_df['reviewRating'] >= 4, 1, 0)) # 1 -> Positive, 0 -> Negative

		nltk_preprocessor = NltkPreprocessor()

		with mp.Pool() as pool:
			reviews_df = reviews_df.assign(cleaned = pool.map(nltk_preprocessor.tokenize, reviews_df['reviewText'])) # Parallel processing
		
		if do_pickle:
			reviews_df.to_pickle('pickled/product_reviews_preprocessed.pickle')

		print('Preprocessing data completed!')
		print('Preprocessing time: ', round(time()-t, 3), 's\n')

	def trainTestSplit(self, reviews_df_preprocessed):
		print('Splitting data using Train-Test split...')
		t = time()
		
		X = reviews_df_preprocessed.iloc[:, -1].values
		y = reviews_df_preprocessed.iloc[:, -2].values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

		print('Splitting data completed!')
		print('Splitting time: ', round(time()-t, 3), 's\n')

		return X_train, X_test, y_train, y_test

	def kFoldSplit(self, reviews_df_preprocessed):
		print('Splitting data using K-Fold Cross Validation...')
		t = time()
		
		X = reviews_df_preprocessed.iloc[:, -1].values
		y = reviews_df_preprocessed.iloc[:, -2].values

		kf = KFold(n_splits = 5, random_state = 42, shuffle = True)
		train_test_indices = kf.split(X, y)

		print('Splitting data completed!')
		print('Splitting time: ', round(time()-t, 3), 's\n')

		return train_test_indices, X, y

	def trainData(self, X_train, y_train, classifier, num_features = 1000000):
		pipeline = []
		model = []

		steps = [
					('vect', TfidfVectorizer(ngram_range = (1,2), use_idf = True, sublinear_tf = True, lowercase = False, stop_words = None, preprocessor = None)),
					('select_best', SelectKBest(score_func = chi2, k = num_features))
				]

		for name, clf in classifier:
			steps.append(('clf', clf))
			pl = Pipeline(steps)
			pipeline.append(pl)

			print('Training data... Classifier ' + str(name))
			t = time()

			model.append((name, pl.fit(X_train, y_train)))

			print('Training data completed!')
			print('Training time: ', round(time()-t, 3), 's\n')

			steps.pop()

		return pipeline, model

	def predictData(self, X_test, model):
		prediction = []

		for name, m in model:
			print('Predicting Test data... Classifier ' + str(name))
			t = time()

			prediction.append((name, m.predict(X_test)))

			print('Prediction completed!')
			print('Prediction time: ', round(time()-t, 3), 's\n')

		return prediction

	def evaluate(self, y_test, prediction):
		clf_accuracy = []
		clf_precision = []
		clf_recall = []
		clf_f1 = []
		clf_roc_auc = []
		clf_cm = []
		clf_cr = []
		
		for name, pred in prediction:
			print('Evaluating results... Classifier ' + str(name))
			t = time()

			clf_accuracy.append(accuracy_score(y_test, pred))
			clf_precision.append(precision_score(y_test, pred))
			clf_recall.append(recall_score(y_test, pred))
			clf_f1.append(f1_score(y_test, pred))
			clf_roc_auc.append(roc_auc_score(y_test, pred))
			clf_cm.append(confusion_matrix(y_test, pred))
			clf_cr.append(classification_report(y_test, pred, target_names = ['negative', 'positive'], digits = 6))

			print('Results evaluated!')
			print('Evaluation time: ', round(time()-t, 3), 's\n')

		return clf_accuracy, clf_precision, clf_recall, clf_f1, clf_roc_auc, clf_cm, clf_cr

	def holdoutStrategy(self, reviews_df_preprocessed, do_pickle, do_train_data):
		print('\nHoldout Strategy...\n')

		if do_train_data:
			X_train, X_test, y_train, y_test = self.trainTestSplit(reviews_df_preprocessed)
			pipeline, model = self.trainData(X_train, y_train, self.clf)

		if do_pickle:
			with open('pickled/features_train.pickle', 'wb') as features_train:
				pickle.dump(X_train, features_train)
			with open('pickled/features_test.pickle', 'wb') as features_test:
				pickle.dump(X_test, features_test)
			with open('pickled/labels_train.pickle', 'wb') as labels_train:
				pickle.dump(y_train, labels_train)
			with open('pickled/labels_test.pickle', 'wb') as labels_test:
				pickle.dump(y_test, labels_test)
			with open('pickled/pipeline_holdout.pickle', 'wb') as pipeline_holdout:
				pickle.dump(pipeline, pipeline_holdout)
			with open('pickled/model_holdout.pickle', 'wb') as model_holdout:
				pickle.dump(model, model_holdout)

		with open('pickled/features_train.pickle', 'rb') as features_train:
			X_train = pickle.load(features_train)
		with open('pickled/features_test.pickle', 'rb') as features_test:
			X_test = pickle.load(features_test)
		with open('pickled/labels_train.pickle', 'rb') as labels_train:
			y_train = pickle.load(labels_train)
		with open('pickled/labels_test.pickle', 'rb') as labels_test:
			y_test = pickle.load(labels_test)
		with open('pickled/pipeline_holdout.pickle', 'rb') as pipeline_holdout:
			pipeline = pickle.load(pipeline_holdout)
		with open('pickled/model_holdout.pickle', 'rb') as model_holdout:
			model = pickle.load(model_holdout)

		prediction = self.predictData(X_test, model)
		clf_accuracy, clf_precision, clf_recall, clf_f1, clf_roc_auc, clf_cm, clf_cr = self.evaluate(y_test, prediction)

		if do_pickle:
			with open('pickled/metrics_cm_holdout.pickle', 'wb') as metrics_cm:
				pickle.dump(clf_cm, metrics_cm)
			with open('pickled/metrics_cr_holdout.pickle', 'wb') as metrics_cr:
				pickle.dump(clf_cr, metrics_cr)

		metrics_list = {
			'Classifier': self.clf_names,
			'Accuracy': clf_accuracy,
			'Precision': clf_precision,
			'Recall': clf_recall,
			'F1-score': clf_f1,
			'ROC AUC': clf_roc_auc
		}

		metrics_df = pd.DataFrame.from_dict(metrics_list)

		for i in range(0, len(self.clf)):
			if i == 0:
				print('======================================================\n')
			print('Evaluation metrics of Classifier ' + self.clf_names[i] + ':')
			print('Confusion Matrix: \n{}\n'.format(clf_cm[i]))
			print('Classification Report: \n{}'.format(clf_cr[i]))
			print('======================================================\n')

		print('Comparison of different metrics for the various Classifiers used:\n')
		print(metrics_df)

		if do_pickle:
			with open('pickled/metrics_dataframe.pickle', 'wb') as df:
				pickle.dump(metrics_df, df)

	def crossValidationStrategy(self, reviews_df_preprocessed, do_pickle):
		print('\nK-Fold Cross Validation Strategy...\n')

		train_test_indices, X, y = self.kFoldSplit(reviews_df_preprocessed)

		accuracy = []
		precision = []
		recall = []
		f1 = []
		roc_auc = []
		cm = []

		for i in range(0, len(self.clf)):
			accuracy.append([])
			precision.append([])
			recall.append([])
			f1.append([])
			roc_auc.append([])
			cm.append(np.zeros((2,2), dtype = 'int32'))

		for train_idx, test_idx in train_test_indices:
			X_train, y_train = X[train_idx], y[train_idx]
			X_test, y_test = X[test_idx], y[test_idx]

			_, model = self.trainData(X_train, y_train, self.clf)
			prediction = self.predictData(X_test, model)
			clf_accuracy, clf_precision, clf_recall, clf_f1, clf_roc_auc, clf_cm, _ = self.evaluate(y_test, prediction)

			for j in range(0, len(self.clf)):
				accuracy[j].append(clf_accuracy[j])
				precision[j].append(clf_precision[j])
				recall[j].append(clf_recall[j])
				f1[j].append(clf_f1[j])
				roc_auc[j].append(clf_roc_auc[j])
				cm[j] += clf_cm[j]

		acc = []
		prec = []
		rec = []
		f1_score = []
		auc = []
		for i in range(0, len(self.clf)):
			if i == 0:
				print('======================================================\n')
			print('Evaluation metrics of Classifier ' + self.clf_names[i] + ':')
			print('Accuracy: {}'.format(np.mean(accuracy[i])))
			print('Precision: {}'.format(np.mean(precision[i])))
			print('Recall: {}'.format(np.mean(recall[i])))
			print('F1-score: {}'.format(np.mean(f1[i])))
			print('ROC AUC: {}'.format(np.mean(roc_auc[i])))
			print('Confusion Matrix: \n{}\n'.format(cm[i]))
			print('======================================================\n')
			acc.append(np.mean(accuracy[i]))
			prec.append(np.mean(precision[i]))
			rec.append(np.mean(recall[i]))
			f1_score.append(np.mean(f1[i]))
			auc.append(np.mean(roc_auc[i]))

		metrics_list = {
			'Classifier': self.clf_names,
			'Accuracy': clf_accuracy,
			'Precision': clf_precision,
			'Recall': clf_recall,
			'F1-score': clf_f1,
			'ROC AUC': clf_roc_auc
		}

		metrics_df = pd.DataFrame.from_dict(metrics_list)

		print('Comparison of different metrics for the various Classifiers used:\n')
		print(metrics_df)

		if do_pickle:
			with open('pickled/metrics_dataframe_kfold.pickle', 'wb') as df_kfold:
				pickle.dump(metrics_df, df_kfold)
