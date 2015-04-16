__author__ = 'Philip'

# What is feature selection?
#Wiki: In machine learning and statistics, feature selection, also known as variable selection, attribute selection or
# variable subset selection, is the process of selecting a subset of relevant features for use in model construction

#Feature selection is different from dimensionality reduction. Both methods seek to reduce the number of attributes in
#the dataset, but a dimensionality reduction method do so by creating new combinations of attributes, where as feature
#selection methods include and exclude attributes present in the data without changing them.

#filter methods
#Example of some filter methods include the Chi squared test, information gain and correlation coefficient scores

#wrapper methods
#stepwise forward backward

#embedded methods
#LASSO, Ridge Regression, Elastic Net

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
X.shape