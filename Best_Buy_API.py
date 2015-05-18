# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:10:42 2015

@author: Philip
"""

import httplib
import json
import os
import pandas as pd
import time
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import textblob as TextBlob
from sklearn.utils import resample

os.getcwd()
os.chdir('C:\Users\Philip\Desktop\Data Science\BestBuy')

api_key = "jbehujrrpbmtx9gzahqbb4aq"

#category id
opt = "/v1/categories"
query ="?format=json"
c = httplib.HTTPConnection("api.remix.bestbuy.com")
c.request('GET','%s%s&apiKey=%s&show=id,name' %(opt,query,api_key))
r = c.getresponse()
data = r.read()

print data

jdata = json.loads(data)['categories']

#category Id TVs
for d in jdata:
    if dict.values(d)[1] == 'TVs':
        tv_id = dict.values(d)[0]
        
#SKU numbers
        
opt = "/v1/products"
query ="(categoryPath.id="
c = httplib.HTTPConnection("api.remix.bestbuy.com")
c.request('GET','%s%s%s)?show=sku&format=json&apiKey=%s' %(opt,query,tv_id,api_key))
r = c.getresponse()
sku = r.read()
sku = json.loads(sku)

sku_total_pages = sku['totalPages']

a = []
for page in range(sku_total_pages):
    page_number = 'page=%s&' %(page+1)
    c.request('GET','%s%s%s)?show=sku&%sformat=json&apiKey=%s' %(opt,query,tv_id,page_number,api_key))
    r = c.getresponse()
    sku = r.read()
    sku = json.loads(sku)
    sku_products = sku['products']
    time.sleep(1)
    for i in sku_products: 
        a.append(dict.values(i))
        
sku_numbers = sum(a,[])

sku_numbers = pd.DataFrame(sku_numbers)

sku_numbers.to_csv('sku_numbers.csv', sep=',', header = False, index = False)


#reviews


b=[]
for i, row in sku_numbers.iterrows():
    sku_number = sku_numbers.iloc[i,0] 
    query ="(sku=%s)?format=json" %(sku_number)
    c = httplib.HTTPConnection("api.remix.bestbuy.com")
    opt = "/v1/reviews"
    c.request('GET','%s%s&apiKey=%s' %(opt,query,api_key))
    r = c.getresponse()
    data = r.read()
    products = json.loads(data)
    print sku_number
    if products['totalPages'] == 0:
        time.sleep(0.5)
        continue
    else:
        products_total_pages = products['totalPages']
        for page in range(products_total_pages):
            print page
            page_number='page=%s&' %(page+1)
            c.request('GET','%s%s&%sapiKey=%s' %(opt,query,page_number,api_key))
            p = c.getresponse()
            print p.status, p.reason
            if p.status == 504:
                c.close()
                c.request('GET','%s%s&%sapiKey=%s' %(opt,query,page_number,api_key))
                p = c.getresponse()
            page_data = p.read()
            productpage = json.loads(page_data)
            productpage = productpage['reviews']
            b.append(productpage)
            time.sleep(0.5)
        c.close()
b = sum(b,[])
reviews_df = pd.DataFrame(b) 
reviews_df.to_pickle('reviews_df')
#reviews_df = pd.read_pickle('reviews_df')

#add new column class based on rating function ( i used rating == 1 and 2 because 1 does not have enough reviews)
def setClass(df):
    if df['rating'] == 5 or df['rating'] == 4:
        return "positive"
    elif df['rating'] == 1 or df['rating'] == 2:
        return "negative"
    else:
        return ""

reviews_df['class'] = reviews_df.apply(setClass, axis = 1)

#reviews_df.to_pickle('reviews_df_spell')
reviews_df_spell = pd.read_pickle('reviews_df_spell')
#reviews_df.to_pickle('reviews_df')   
reviews_df = pd.read_pickle('reviews_df')

#exploratory analysis
reviews_df.groupby('rating').size()
#rating
#1          1066
#2           842
#3          2824
#4         21199
#5         50792
reviews_df.groupby('class').size()
#class
#             2824
#negative     1908
#positive    71991

#undersample dataset
num_negative = sum(reviews_df['class'] == 'negative')
positive_sample_indx = list(resample(reviews_df[reviews_df['class']=='positive'].index.tolist(), n_samples = int(num_negative), random_state = 0))
positive_sample = reviews_df.loc[positive_sample_indx,:]
negative_sample = reviews_df[reviews_df['class'] == 'negative']
undersample = positive_sample.append(negative_sample)


#replace puncs with spaces and list each word, removed numbers and words with numbers, spell check with textblob
def preprocess(df,column):
    #df[column] = df[column].apply(lambda x: str(TextBlob.TextBlob(x).correct()))
    #df[column] = df[column].apply(lambda x : re.sub('\W',' ',x).strip())
    #df[column] = df[column].apply(lambda x : re.sub(r'\w*\d\w*', '', x).strip())
    df[column] = df[column].apply(lambda x : re.sub('\d',' ',x).strip())

#scikit
def naive_bayes(df,column):
    reviews_pn = df[df['class'].isin(['positive','negative'])]
    comments = list(reviews_pn[column].values)
    classes = list(reviews_pn['class'].values)
    
    # preprocess creates the term frequency matrix for the review data set
    stop = stopwords.words('english')
    count_vectorizer = CountVectorizer(stop_words = stop, ngram_range=(1,3))
    comments1 = count_vectorizer.fit_transform(comments)
    tfidf_comments = TfidfTransformer(use_idf=True).fit_transform(comments1)
    
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(tfidf_comments,classes,test_size=0.4,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    
    print classification_report(target_test,predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_test,predicted))
    
    most_informative_feature_for_binary_classification(count_vectorizer,classifier,n=20)
    
    #predict on unknown
    reviews_nc = reviews_df[reviews_df['class'] == '']
    comments_nc = list(reviews_nc[column].values)
    comments_nc1 = count_vectorizer.transform(comments_nc)    
    tfidf_comments_nc = TfidfTransformer(use_idf=True).fit_transform(comments_nc1)    
    new_predicted = classifier.predict(tfidf_comments_nc)
    
    print "negative = %s" %sum(new_predicted == 'negative')
    print "positive = %s" %sum(new_predicted == 'positive')
    

#top features
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=10):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print class_labels[0], coef, feat

    print

    for coef, feat in reversed(topn_class2):
        print class_labels[1], coef, feat
        
#final output


def main(df, column):
    preprocess(df,column)
    naive_bayes(df,column)

main(undersample, 'comment')
main(undersample,'title')






