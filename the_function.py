import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

df = pd.read_csv('Hotel_Reviews.csv')

#learning how the data set looks like

#print(df.head())
#print(df.describe())
print(df.info())
#print(df.isnull().sum())
print(df.iloc[1])
print(df.iloc[2000,0])

#simplyfying the "United Kingdom" into "UK"
df.Hotel_Address = df.Hotel_Address.str.replace("United Kingdom", "UK")

#creating a new column with country (separating from the hotel address)
df['country'] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
print(df)

value_counts = df['country'].value_counts()
print(value_counts)

#dropping unnecessary columns
columns_to_drop = ['Reviewer_Nationality', 'Negative_Review', 'Review_Total_Negative_Word_Counts', 'Positive_Review', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score', 'days_since_review', 'lat', 'lng']

df.drop(columns_to_drop, axis=1, inplace=True)

#understanding the 'Tags' column
value_counts = df['Tags'].value_counts()
print(value_counts)

# Function to correct the format of Tags column
def correction(string):
    string = string[0]
    if type(string) != list:
        return "".join(literal_eval(string))
    else:
        return string
    
# the hotel recommendation system machine
def recommend_hotel(location, description):
    description = description.lower()
    description_tokens = word_tokenize(description)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    filtered = {word for word in description_tokens if word not in stop_words}
    filtered_set = {lemmatizer.lemmatize(word) for word in filtered}
    
    country = df[df['country'] == location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    
    cos = []
    for i in range(country.shape[0]):
        temp_tokens = word_tokenize(country["Tags"][i].lower())
        temp_set = {word for word in temp_tokens if word not in stop_words}
        temp2_set = {lemmatizer.lemmatize(word) for word in temp_set}
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    
    country['similarity'] = cos
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Average_Score', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    
    return country[["Hotel_Name", "Average_Score", "Total_Number_of_Reviews", "Hotel_Address"]].head(10)


#function to run the hotel recommendation system
def run():
    country = input("Which country are you going to stay in (UK, France or Netherlands?\t:")
    description = input("What is the purpose of your travel?\t:")
    return recommend_hotel(country, description)

run() 
