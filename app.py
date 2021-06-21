from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class GetData:
    """ Class to get all the pickled model and dataset
    """
    def __init__(self):
        self.__edaFinal = pd.read_pickle("./data/final_eda", compression="zip")
        self.__userMappingData = self.__edaFinal.drop_duplicates(
            subset=["id"], keep="first"
        )[["id", "name"]]
        self.__sentimentData = pd.read_pickle("./data/sentiment_stats", compression="zip")
        self.__userRatingData = pd.read_pickle('./data/user_rating', compression='zip')
    
    def getCleanedData(self):
        """ To get the final cleaned dataset
        """
        return self.__edaFinal
    
    def getMappedData(self):
        """ To get the user mapping data
        """
        return self.__userMappingData
    
    def getSentimentData(self):
        """ To get the sentiment data
        """
        return self.__sentimentData
    
    def getUserRatingData(self):
        """ To get the user rating data
        """
        return self.__userRatingData


app = Flask(__name__)

# Get all the dataframe 
dataObject = GetData()
mappedData = dataObject.getMappedData()
userRatingData = dataObject.getUserRatingData()
cleanData = dataObject.getCleanedData()
sentimentData = dataObject.getSentimentData()


def checkProductSentiment(productList, productsSentiments):
    """ Function to calculate final product list based on sentiment
    """
    productPercent = {}
    for id in productList:
        filteredProduct = productsSentiments[productsSentiments['id']==id]
        percentPositive = filteredProduct['Prediction'].sum()/len(filteredProduct)
        productPercent[id]=percentPositive
    productPercentAsc =sorted(productPercent.items(), key=lambda x: x[1])
    finalprodList = [i [0] for i in productPercentAsc[::-1][:5]]
    return finalprodList

def getRecommendedProduct(username, map_df):
    """ Function to get the top 5 recommended products and render it
    """
    df_final = pd.DataFrame()
    df_final_5 = pd.DataFrame()
    isError = None
    userInfo = None

    if username not in userRatingData.index:
        isError = 'Data Not Available'
    else:
        df_final = userRatingData.loc[username].sort_values(ascending=False)[0:20]
        df_final = pd.concat({"id": pd.Series(list(df_final.index)),
                            "probScore": pd.Series(list(df_final.values))},axis=1)
        df_final = pd.merge(df_final, map_df, left_on='id', right_on='id', how = 'left')
        userIds = list(df_final['id'])
        final_5 = checkProductSentiment(userIds, sentimentData)
        df_final_5= df_final[df_final['id'].isin(final_5)][['id', 'name']]
        userInfo = username

    return render_template('index.html', username=userInfo, data=[df_final_5.to_html(classes='prediction', index=False)], error=isError, users=None, titles=[])

@app.route('/')
@app.route('/index')
def index():
    all_users = np.random.choice(cleanData.reviews_username.unique(), size=5)
    return render_template('index.html', username=None, error=None, users=all_users)

'''
    Submit request on the form. 
'''
@app.route('/topProducts', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['name']
        return getRecommendedProduct(user, mappedData)

if __name__ == "__main__":
    app.run()
