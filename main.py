# -*- coding: utf-8 -*-
# @Author: pranit
# @Date:   2018-05-16 11:00:08
# @Last Modified by:   pranit
# @Last Modified time: 2018-05-17 05:28:57

from sentiment_analyzer import SentimentAnalyzer
from data_visualizer import DataVisualizer
from utilities import Utility 

import pickle
import pandas as pd

def analyzeVisualize(sentiment):
	with open('pickled/pipeline_holdout.pickle', 'rb') as pipeline_holdout:
		pipeline = pickle.load(pipeline_holdout)
	with open('pickled/metrics_cm_holdout.pickle', 'rb') as metrics_cm:
		clf_cm = pickle.load(metrics_cm)
	with open('pickled/metrics_cr_holdout.pickle', 'rb') as metrics_cr:
		clf_cr = pickle.load(metrics_cr)
	with open('pickled/metrics_dataframe.pickle', 'rb') as df:
		metrics_df = pickle.load(df)
	with open('pickled/metrics_dataframe_kfold.pickle', 'rb') as df:
		metrics_df_kfold = pickle.load(df)

	clf_svc = pipeline[2]
	clf_names = sentiment.clf_names
	labels = ['negative', 'positive']

	visualizer = DataVisualizer()

	for cm, cr, name in zip(clf_cm, clf_cr, clf_names):
		visualizer.plotConfusionMatrix(cm, labels, name)
		visualizer.plotClassificationReport(cr, labels, name)
	
	metrics_df.rename(columns = {"Accuracy": "value_Accuracy", "Precision": "value_Precision", "Recall": "value_Recall", "F1-score": "value_F1-score", "ROC AUC": "value_ROC AUC"}, inplace = True)
	metrics_df['id'] = metrics_df.index
	metrics_df_long = pd.wide_to_long(metrics_df, stubnames = 'value', i = 'id', j = 'id_m', sep = '_', suffix = '\w')
	metrics_df_long['Metrics'] = metrics_df_long.index.get_level_values('id_m')
	visualizer.plotClassifierPerformanceComparison(metrics_df_long, clf_names, 'Holdout')
	
	metrics_df_kfold.rename(columns = {"Accuracy": "value_Accuracy", "Precision": "value_Precision", "Recall": "value_Recall", "F1-score": "value_F1-score", "ROC AUC": "value_ROC AUC"}, inplace = True)
	metrics_df_kfold['id'] = metrics_df_kfold.index
	metrics_df_kfold_long = pd.wide_to_long(metrics_df_kfold, stubnames = 'value', i = 'id', j = 'id_m', sep = '_', suffix = '\w')
	metrics_df_kfold_long['Metrics'] = metrics_df_kfold_long.index.get_level_values('id_m')
	visualizer.plotClassifierPerformanceComparison(metrics_df_kfold_long, clf_names, 'K-Fold')
	
	util = Utility()

	data = util.classifiersVsFeatures()
	colors = ['blue', 'yellow', 'red', 'green']
	visualizer.plotClassifiersVsFeatures(data, clf_names, colors)

	top_features = util.showTopFeatures(clf_svc, n = 30)
	print('The 30 most informative features for both positive and negative coefficients:\n')
	print(top_features)

if __name__ == "__main__":

	do_pickle = False

	# sentiment = SentimentAnalyzer()

	# senitment.getInitialData('datasets/product_reviews.json', do_pickle)
	# reviews_df = pd.read_pickle('pickled/product_reviews.pickle')

	# sentiment.preprocessData(reviews_df, do_pickle)
	# reviews_df_preprocessed = pd.read_pickle('pickled/product_reviews_preprocessed.pickle')
	# print(reviews_df_preprocessed.isnull().values.sum()) # Check for any null values

	# sentiment.crossValidationStrategy(reviews_df_preprocessed, do_pickle)
	# sentiment.holdoutStrategy(reviews_df_preprocessed, do_pickle)

	# analyzeVisualize(sentiment)
	
	with open('pickled/model_holdout.pickle', 'rb') as model_holdout:
		model = pickle.load(model_holdout)

	model_svc = model[2][1]
	
	print('\nEnter your review:')
	user_review = input()
	verdict = 'Positive' if model_svc.predict([user_review]) == 1 else 'Negative'
	print('\nPredicted sentiment: '+ verdict)

