from Utils import *
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDFModel(data_set):
    Tfidf_Vectorizer = TfidfVectorizer(min_df=0) #ignore terms in vocabulary having lower frequency thanm threshold
    Vectorized_Title=Tfidf_Vectorizer.fit_transform(data_set['title'])
    apparel_id=int(input("Enter the Apparel ID :\n"))
    num_recc=15
    pairwise_dist = pairwise_distances(Vectorized_Title,Vectorized_Title[apparel_id])
    recc_indices = np.argsort(pairwise_dist.flatten())[0:num_recc]
    recc_pdists = np.sort(pairwise_dist.flatten())[0:num_recc]
    df_recc_indices = list(data_set.index[recc_indices])
    for i in range(0,len(recc_indices)):
        GetRecomendations(recc_indices[i],data_set['title'].loc[df_recc_indices[0]],data_set['title'].loc[df_recc_indices[i]],data_set['medium_image_url'].loc[df_recc_indices[i]],'tfidf')
        print('ASIN :',data_set['asin'].loc[df_recc_indices[i]])
        print('TITLE :',data_set['title'].loc[df_recc_indices[i]])
        print('BRAND :',data_set['brand'].loc[df_recc_indices[i]])
        print('Similarity Index(Euclidian Similarity) :',recc_pdists[i])
        print("_"*70)
        