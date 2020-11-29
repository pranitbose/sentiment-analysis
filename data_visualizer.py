import string
import re

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns

class DataVisualizer:
	def plotConfusionMatrix(self, cm, labels, clf_name):
		fig, ax = plt.subplots()
		sns.heatmap(cm, annot=True, ax = ax, fmt = 'g')

		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('Actual labels')
		ax.set_title('Confusion Matrix of {} Classifier'.format(clf_name))
		ax.xaxis.set_ticklabels(labels)
		ax.yaxis.set_ticklabels(labels, rotation = 0)
		plt.tight_layout()
		fig.savefig('plots/cm_{}.png'.format(clf_name.lower().replace(' ', '_')))
		plt.close()

	def plotClassificationReport(self, cr, labels, clf_name):
		cr_mat = []
		allowed_labels = ['negative', 'positive', 'weighted avg']

		lines = cr.split('\n')
		for line in lines[2 : -1]:
			line = line.strip()
			if line == '':
				continue
			row = re.split(r'\s{2,}', line)
			if row[0] not in allowed_labels:
				continue
			row_data = []
			row_data.append(float(row[-4]))
			row_data.append(float(row[-3]))
			row_data.append(float(row[-2]))
			row_data.append(float(row[-1]))
			cr_mat.append(row_data)

		xlabels = ['precision', 'recall', 'f1-score', 'support']
		ylabels = labels + ['weighted avg']

		fig, ax = plt.subplots()
		sns.heatmap(cr_mat, annot = True, ax = ax, fmt = 'g')

		ax.set_xlabel('Metrics')
		ax.set_ylabel('Classes')
		ax.set_title('Classification Report of {} Classifier'.format(clf_name))
		ax.xaxis.set_ticklabels(xlabels)
		ax.yaxis.set_ticklabels(ylabels, rotation = 0)
		plt.tight_layout()
		fig.savefig('plots/cr_{}.png'.format(clf_name.lower().replace(' ', '_')))
		plt.close()

	def plotClassifierPerformanceComparison(self, metrics_df, clf_names, strategy):
		fig, ax = plt.subplots()
		sns.barplot(x = 'Metrics', y = 'value', data = metrics_df, ax = ax, hue = 'Classifier')

		ax.set_xlabel('Evaluation Metrics')
		ax.set_ylabel('Classifier\'s performance')
		ax.set_title('Overall Comparison of Classifier\'s Performance (' + strategy + ')')
		pos = ax.get_position()
		ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
		plt.legend(bbox_to_anchor = (1, 0.5), loc = 'best')
		plt.tight_layout()
		if strategy == 'K-Fold':
			fig.savefig('plots/classifiers_vs_metrics_kfold.png')
		else:
			fig.savefig('plots/classifiers_vs_metrics.png')
		plt.close()

	def plotClassifiersVsFeatures(self, data, clf_names, colors):
		fig, ax = plt.subplots()
		lines = []
		for d, c, clf_name in zip(data, colors, clf_names):
			sns.pointplot(x = 'x', y = 'y', data = d, ax = ax, color = c)
			lines.append(mpatches.Patch(color = c, label = clf_name))

		ax.legend(handles = lines, bbox_to_anchor=(1, 0.5), loc = 'best')
		ax.set_xlabel('K-Best Features')
		ax.set_ylabel('Classification Accuracy Scores')
		ax.set_title('Comparison of Classifier\'s Performance over Selected Features')
		fig.savefig('plots/classifiers_vs_features.png')
		plt.close()
