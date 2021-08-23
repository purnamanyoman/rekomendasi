# Utility functions
import requests
import seaborn
from io import BytesIO
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib import gridspec
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

def GetRecomendations(apparel_id,title1,title2,url,model_name):
    temp_title=title2
    vector1=convert_text_to_vec(title1) #words : counts
    vector2=convert_text_to_vec(title2)
    PlotHeatMapImage(apparel_id,vector1,vector2,url,temp_title,model_name)

def convert_text_to_vec(title):
    words=title.split()
    vector=Counter(words)
    return vector

def DisplayImageFromUrl(url,axes,fig):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    plt.imshow(img)

def PlotHeatmap(words_of_title,occurences,model_labels,url,title):
    #divide figure to two parts
    grid_s=gridspec.GridSpec(2,2,width_ratios=[4,1],height_ratios=[4,1])
    fig = plt.figure(figsize=(25,3))
    #heatmap---> commonly occured words in title2
    axes = plt.subplot(grid_s[0])
    axes = seaborn.heatmap(np.array([occurences]),annot=np.array([occurences]))
    axes.set_xticklabels(words_of_title)
    axes.set_title(title)
    
    #plotting image of item
    axes=plt.subplot(grid_s[1])
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    
    DisplayImageFromUrl(url,axes,fig)
    plt.show()
        
def PlotHeatMapImage(doc_id, vec1, vec2, url, text, model):
    # the common words contribute to distance/similarity index
    intersection = set(vec1.keys()) & set(vec2.keys()) 
    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    keys = list(vec2.keys()) 
    values = [vec2[x] for x in vec2.keys()] #occurences of the words present in interection

    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        data_set=pd.read_pickle('Pickles/Preprocessed_Data_version3')
        Tfidf_Vectorizer = TfidfVectorizer(min_df=0) #ignore terms in vocabulary having lower frequency thanm threshold
        Vectorized_Title=Tfidf_Vectorizer.fit_transform(data_set['title'])
        labels = []
        for x in vec2.keys():
            if x in  Tfidf_Vectorizer.vocabulary_:
                labels.append(Vectorized_Title[doc_id, Tfidf_Vectorizer.vocabulary_[x]]) # VectorizedTitles[apparel_id,words_in_corpus] = tfidf value
            else:
                labels.append(0)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            if x in  idf_title_vectorizer.vocabulary_:
                labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)

    PlotHeatmap(keys, values, labels, url, text)
