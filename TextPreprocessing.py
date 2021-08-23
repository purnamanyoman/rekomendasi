import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def TextPreprocessing(data_set):
    data_set=FilterStopWords(data_set)
    data_set.to_pickle("Pickles/Preprocessed_Data_version3")
    return data_set

def FilterStopWords(data_set):
    stop_words = set(stopwords.words('english'))
    for row_index,row in data_set.iterrows():
        title=row['title']
        new_title=""
        for word in title.split():
            #remove special characters
            word=("".join(letter for letter in word if letter.isalnum())).lower()
            if word not in stop_words:
                new_title+= word+ " "
        data_set['title'][row_index] = new_title
    return data_set
