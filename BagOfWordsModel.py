import re
from io import BytesIO
from Utils import *
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from PIL import Image
import numpy as np

def BagOfWordsModel(data_set):
    #we use the 'title' feature for recommendation
    VectorizedTitles=Vectorize(data_set)
    apparel_id=int(input("Enter the Apparal ID : \n"))
    num_recc=20
    pairwise_dist = pairwise_distances(VectorizedTitles,VectorizedTitles[apparel_id])
    recc_indices = np.argsort(pairwise_dist.flatten())[0:num_recc]
    recc_pdists = np.sort(pairwise_dist.flatten())[0:num_recc]
    df_recc_indices = list(data_set.index[recc_indices])
    for i in range(0,len(recc_indices)):
        GetRecomendations(recc_indices[i],data_set['title'].loc[df_recc_indices[0]],data_set['title'].loc[df_recc_indices[i]],data_set['medium_image_url'].loc[df_recc_indices[i]],'bag_of_words')
        print('ASIN :',data_set['asin'].loc[df_recc_indices[i]])
        print('TITLE :',data_set['title'].loc[df_recc_indices[i]])
        print('BRAND :',data_set['brand'].loc[df_recc_indices[i]])
        print('Similarity Index(Euclidian Similarity) :',recc_pdists[i])
        print("_"*70)
        
def Vectorize(data_set):
    Count_Vectorizer=CountVectorizer()
    VectorizedTitles=Count_Vectorizer.fit_transform(data_set['title'])
    return VectorizedTitles
