# Analysis-of-Customer-Reviews-Using-Big-Data
Built a sentiment analysis platform processing 2M+ customer reviews, using Python, NLP libraries like NLTK/TextBlob, and big data tools like Pandas/NumPy
<a id="section-one"></a>
# Introduction

**Every day, we encounter numerous products on digital platforms, presenting us with various choices within the same category. Choosing a product can be daunting, but customer reviews come to our aid. When customers purchase a product, they often leave a rating and review detailing their experience, helping others make informed decisions. However, manually assessing each review for positivity or negativity can be time-consuming. Fortunately, advancements in Natural Language Processing (NLP) and artificial intelligence have greatly simplified this process.**

## What is sentiment analysis?

**Sentiment Analysis is a technique to classify text that determines whether a message expresses a positive, negative, or neutral sentiment. Businesses are interested in understanding customer sentiments because customers frequently share their thoughts and emotions. Analyzing customer feedback automatically through technology enables brands to listen closely to their customers, thereby aligning their products and services with customer needs.**

## Problem Statement

**The objective is to develop a platform that provides concise summaries of critical business information, such as recent reviews, overall ratings, sentiment distribution, and trending keywords. Currently, businesses must manually read and interpret a large volume of customer reviews, comments, and concerns, which requires significant effort, time, and resources.**

## Project Objectives

1. Preprocess and Clean Reviews
2. Generate Stories and Visualize Insights from Reviews
3. Extract Features from Processed Reviews
4. Build a Sentiment Analysis Model

## Import Libraries

```python
# Basic libraries
import pandas as pd 
import numpy as np 

# NLTK libraries
import nltk
import re
import string
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning libraries
import sklearn 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn import preprocessing 

# Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Other miscellaneous libraries
from scipy import interp
from itertools import cycle
from collections import defaultdict
from collections import Counter
from imblearn.over_sampling import SMOTE
