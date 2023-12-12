import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hotel_reviews.csv')

#histogram
#df.hist(figsize=(10, 8), bins=20)
#plt.show()

# Box plots
#sns.boxplot(x='Average_Score', data=df)
#plt.show()

# Bar plots for categorical variables
#sns.countplot(x='Average_Score', data=df)
#plt.show()

df_new = df.drop('Hotel_Address', axis=1)
print(df_new)

# Correlation matrix
corr_matrix = df.corr()