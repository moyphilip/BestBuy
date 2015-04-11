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

os.getcwd()
os.chdir('C:\Users\Philip\Desktop\Data Science\Best Buy')

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
#reviews_df.to_pickle('reviews_df')
#reviews_df = pd.read_pickle('reviews_df')

#add new column class based on rating function ( i used rating == 1 and 2 because 1 does not have enough reviews)
def setClass(df):
    if df['rating'] == 5:
        return "positive"
    elif df['rating'] == 1 or df['rating'] == 2:
        return "negative"
    else:
        return ""

reviews_df['class'] = reviews_df.apply(setClass, axis = 1)

#replace puncs with spaces and list each word, need to replace numbers and words with numbers
reviews_df['comment'] = reviews_df['comment'].str.replace('[^\w\s]',' ')
reviews_df['comment'] = reviews_df['comment'].apply(lambda x : re.sub(r'\w*\d\w*', '', x).strip())
#reviews_df.to_pickle('reviews_df')   
reviews_df = pd.read_pickle('reviews_df')

#spell check
from textblob import TextBlob

#scikit try
reviews_pn = reviews_df[reviews_df['class'].isin(['positive','negative'])]
comments = list(reviews_pn['comment'].values)
classes = list(reviews_pn['class'].values)
class_names = ['negative','positive']

# preprocess creates the term frequency matrix for the review data set
stop = stopwords.words('english')
count_vectorizer = CountVectorizer(analyzer =u'word',stop_words = stop, ngram_range=(1, 3))
comments = count_vectorizer.fit_transform(comments)
tfidf_comments = TfidfTransformer(use_idf=True).fit_transform(comments)


# preparing data for split validation. 60% training, 40% test
data_train,data_test,target_train,target_test = cross_validation.train_test_split(tfidf_comments,classes,test_size=0.4,random_state=43)
classifier = BernoulliNB().fit(data_train,target_train)
predicted = classifier.predict(data_test)
    
print classification_report(target_test,predicted)
print "The accuracy score is {:.2%}".format(accuracy_score(target_test,predicted))

#top 10 features
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[0])[:20]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)





#first try
reviews_positive = reviews_df[reviews_df['class'] == 'positive']
reviews_negative = reviews_df[reviews_df['class'] == 'negative']

#reviews split by word
reviews_positive['comment']= reviews_positive['comment'].str.lower().str.split()
reviews_negative['comment'] = reviews_negative['comment'].str.lower().str.split()

#remove stop words from comment
from nltk.corpus import stopwords
stop = stopwords.words('english')
reviews_positive['comment'] = reviews_positive['comment'].apply(lambda x: [item for item in x if item not in stop])
reviews_negative['comment'] = reviews_negative['comment'].apply(lambda x : [item for item in x if item not in stop])

#reviews_positive.to_pickle('reviews_positive')
#reviews_negative.to_pickle('reviews_negative')

#reviews_positive = pd.read_pickle('reviews_positive')
#reviews_negative = pd.read_pickle('reviews_negative')


[reviews_positive['comment']
