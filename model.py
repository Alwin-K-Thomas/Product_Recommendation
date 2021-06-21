import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import datetime
import string
import unicodedata
from bs4 import BeautifulSoup
import pickle
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import *
from sklearn import *

from imblearn.over_sampling import SMOTE



recommend_df = pd.read_csv("datasets/sample30.csv")
recommend_df.drop(
    columns=['reviews_userProvince', 'reviews_userCity'],
    axis=1,
    inplace=True
)
recommend_df = recommend_df[recommend_df['reviews_username'].notna()]
recommend_df = recommend_df[recommend_df['reviews_username'].notna()]
recommend_df = recommend_df[recommend_df['user_sentiment'].notna()]
recommend_df['reviews_title']=  recommend_df['reviews_title'].fillna(' ')
recommend_df['reviews_didPurchase'].fillna(False , inplace=True)
recommend_df['reviews_doRecommend'].fillna(False , inplace=True)
recommend_df['manufacturer'].fillna('Unknown manufacturer', inplace=True)
recommend_df.drop_duplicates(
    subset=['id', 'reviews_username', 'reviews_date'],
    keep='first',
    inplace=True
)
recommend_df.drop(columns=['reviews_date'], axis=1, inplace=True)

map_products = recommend_df.drop_duplicates(subset='id',keep='first')[['id','name','categories']]
recommend_df_master = recommend_df[['id','reviews_username','name','reviews_rating','user_sentiment']].copy()
recommend_df_master['reviews'] = recommend_df['reviews_title'] + " " + recommend_df['reviews_text']
recommend_df_master['user_sentiment'] = recommend_df_master['user_sentiment'].map(lambda x: 1 if x.lower() == 'positive' else 0)
recommend_df_master['sentiment'] = recommend_df_master['user_sentiment']
recommend_df_master.drop(columns=['user_sentiment'], axis=1, inplace=True)

raw_tokens=len([w for t in (recommend_df_master["reviews"].apply(word_tokenize)) for w in t])

punc_ext = string.punctuation
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def lower_data(col):
    lower_col = col.apply(lambda x: x.lower())
    return lower_col

def remove_punctuation(text):
    return text.translate(text.maketrans('', '', punc_ext))

def remove_special_characters(col):
    no_special_characters = col.replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    return no_special_characters

def remove_whitespace(col):
    # replace more than 1 space with 1 space
    merged_spaces = col.apply(lambda x: x.replace(r"\s\s+",' '))
    # delete beginning and trailing spaces
    trimmed_spaces = merged_spaces.apply(lambda x: x.strip())
    return trimmed_spaces

def remove_website_links(col):
    no_website_links = col.str.replace(r"http\S+", "")
    return no_website_links

def remove_numbers(col):
    removed_numbers = col.apply(lambda x: x.replace(r'\d+',''))
    return removed_numbers

def remove_emails(col):
    no_emails = col.apply(lambda x: x.replace(r"\S*@\S*\s?", ""))
    return no_emails

def tokenize_col(col):
    tokenized_col = col.apply(lambda x: tokenizer.tokenize(str(x)))
    return tokenized_col

def remove_stopwords(text): 
    words = [w for w in text if w not in stopwords.words('english')]
    return words

# lower text
recommend_df_master['reviews'] = lower_data(recommend_df_master['reviews'])
# Remove whitespaces
recommend_df_master['reviews'] = remove_whitespace(recommend_df_master['reviews'])
# Remove punctuations
recommend_df_master['reviews'] = recommend_df_master['reviews'].apply(lambda line: remove_punctuation(line))
# Remove special characters
recommend_df_master['reviews'] = remove_special_characters(recommend_df_master['reviews'])
# Remove website links
recommend_df_master['reviews'] = remove_website_links(recommend_df_master['reviews'])
# Remove emails
recommend_df_master['reviews'] = remove_emails(recommend_df_master['reviews'])
# Remove numbers
recommend_df_master['reviews'] = remove_numbers(recommend_df_master['reviews'])

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def word_lemmatizer(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    lem_text = [lemmatizer.lemmatize(i) for i in words]
    lem_text = " ".join(lem_text)
    
    return lem_text

recommend_df_master['reviews'] = [ word_lemmatizer(i) for i in recommend_df_master['reviews']]

def stemming_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    stem_text = [stemmer.stem(i) for i in words]
    stem_text = " ".join(stem_text)
    
    return stem_text

recommend_df_master['reviews'] = [ stemming_text(i) for i in recommend_df_master['reviews']]

X = recommend_df_master['reviews']
y = recommend_df_master['sentiment']

tfidf_vec = TfidfVectorizer(
    max_features=None, 
    lowercase=True, 
    analyzer='word', 
    stop_words= stop_words, 
    ngram_range=(1, 1)
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=seed)

## Applying TF-IDF to extract train, test feature set
tfidf_vec.fit(X_train)
X_train = tfidf_vec.transform(X_train)
X_test = tfidf_vec.transform(X_test)

## Applying SMOTE for imbalance data
sm = SMOTE()
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
X_test_smote, y_test_smote = sm.fit_sample(X_test, y_test)

## Logistic Regression
model = LogisticRegression(penalty="l2", random_state=seed, max_iter=500)

## Model fit
model.fit(X_train_smote, y_train_smote)

## Model Prediction
y_pred = model.predict(X_test_smote)

## Accuracy
train_accuracy = model.score(X_train_smote, y_train_smote)
test_accuracy = model.score(X_test_smote, y_test_smote)

"""## Save the ML Model"""

# Save the model as a pickle in a file 
joblib.dump(model, 'models/model.pkl')
# Save the TF-IDF model as a pickle in a file 
joblib.dump(tfidf_vec, 'models/vectorizer.pkl')

df_recommend = pd.DataFrame(data = recommend_df_master[['reviews_username','id','reviews_rating']])
df_recommend = df_recommend.groupby(['reviews_username','id'],as_index=False).agg({'reviews_rating': pd.Series.mode,'reviews_rating':np.mean})
## Changing column names
df_recommend['username'] = df_recommend['reviews_username']
df_recommend['product'] = df_recommend['id']
df_recommend['rating'] = round(df_recommend['reviews_rating'],1)
## Dropping the old columns
df_recommend = df_recommend.drop(['reviews_username','id','reviews_rating'],axis=1)

train, test = train_test_split(df_recommend, test_size=0.2, random_state=42)
df_product_features = train.pivot(
    index='username',
    columns='product',
    values='rating'
).fillna(0)

dummy_train = train.copy()
dummy_test = test.copy()
## Changing the rating value to '0' if the user provided rating
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)
## Changing the rating value to '1' if the user provided rating
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_train = dummy_train.pivot(
    index='username',
    columns='product',
    values='rating'
).fillna(1)

dummy_test = dummy_test.pivot(
    index='username',
    columns='product',
    values='rating'
).fillna(0)

product_features = train.pivot(
    index='username',
    columns='product',
    values='rating'
).T

mean = np.nanmean(product_features, axis=1)
df_subtracted = (product_features.T-mean).T



# User Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
item_correlation[item_correlation<0]=0

item_predicted_ratings = np.dot((product_features.fillna(0).T),item_correlation)
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.iloc[1].sort_values(ascending=False)[0:20]

final_pred_df = pd.DataFrame()
for i in np.arange(0,20257):  
    ## Getting the top 15 products for each user
    df = item_final_rating.iloc[i].sort_values(ascending=False)[:20]

    ## Removing the products which dont have any weightage calculated
    product_id = np.array(df.index[df.values != 0])
    user_name = item_final_rating.iloc[i].name

    ##Creating and storing in data frame at user_name and product
    sorted_user_predictions = pd.DataFrame({'product_id':product_id,'user_name':user_name})
    sorted_user_predictions.assign(product_id=sorted_user_predictions.product_id.str.split(",")).explode("product_id")

    sorted_user_predictions.set_index('product_id')
    final_pred_df = final_pred_df.append(sorted_user_predictions)


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def getSentimentData(model, X):
    pred=model.predict(X)
    prediction = model.predict_proba(X)
    df_pred = pd.DataFrame()
    df_pred['id'] = df['id']
    df_pred['name'] = df['name']
    df_pred['reviews'] = df['reviews']
    df_pred_prob = pd.DataFrame(prediction, columns=['Negative','Positive'])
    df_pred['max_prob'] = df_pred_prob[['Negative','Positive']].max(axis=1)
    df_pred['max_prob_class'] = df_pred_prob.idxmax(axis=1)
    df_pred['Prediction'] = pred
    return df_pred

def checkProductSentiment(productList, productsSentiments):
    productPercent = {}
    for id in productList:
        filteredProduct = productsSentiments[productsSentiments['id']==id]
        percentPositive = filteredProduct['Prediction'].sum()/len(filteredProduct)
        productPercent[id]=percentPositive
    productPercentAsc =sorted(productPercent.items(), key=lambda x: x[1])
    finalprodList = [i [0] for i in productPercentAsc[::-1][:5]]
    return finalprodList

def getRecommendedProduct(username, productMapping, productsSentiments, final_rating):
    df_final = final_rating.loc[username].sort_values(ascending=False)[0:20]
    df_final = pd.concat({"id": pd.Series(list(df_final.index)),
                        "probScore": pd.Series(list(df_final.values))},axis=1)
    df_final = pd.merge(df_final, productMapping, left_on='id', right_on='id', how = 'left')
    productList = list(df_final['id'])
    #Get the list of 20 products to check their sentiment
    final5 = checkProductSentiment(productList,productsSentiments)
    df_final5 = df_final[df_final['id'].isin(final5)]
    print('Already Bought' + df[df['reviews_username']==username]['name'].head())
    print('All 20 recommendation')
    print(df_final.head())
    print('Top 5')
    print(df_final5.head())

X_transformed_LR_tfidf=vectorizer.transform(df['reviews'].tolist())
sentimentStats = getSentimentData(model, X_transformed_LR_tfidf)

# Pivot the train ratings' dataset into matrix format
t_pivot = df_recommend[['username','product','rating']]

df_pivot_user = t_pivot.pivot(
    index='username',
    columns='product',
    values='rating'
).fillna(0)
dummy_train = t_pivot.copy()
dummy_train = dummy_train.pivot(
    index='username',
    columns='product',
    values='rating'
).fillna(1)

user_correlation = 1 - pairwise_distances(df_pivot_user, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
user_correlation[user_correlation < 0]=0
user_predicted_ratings = np.dot(user_correlation, df_pivot_user.fillna(0))
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

user_final_rating.to_pickle('user_rating',compression='zip')
sentimentStats.to_pickle('sentiment_stats',compression='zip')
