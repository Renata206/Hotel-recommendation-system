import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hotel_reviews.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())

#dropping lat and lng columns, because there are quite many missing values (3268 out of 515738)
missing_values = df[df['lat'].isnull()]
print(missing_values)
