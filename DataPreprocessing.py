import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Data_Preprocessing(data_set):
    # descending order of  of Null values in prices>color>brand
    data_set=RemoveNullValues(data_set)
    # Recommendtaion based on different size or color not a good recommendation
    data_set=RemoveDuplicateItemsAfterSorting(data_set)
    print(data_set.shape[0])
    data_set.to_pickle('Pickles/Preprocessed_Data_version1')
    data_set=RemoveDuplicateItemsNonAdjacent(data_set)
    data_set.to_pickle('Pickles/Preprocessed_Data_version2')
    return data_set

 
def RemoveNullValues(data_set):
    hav_null=['formatted_price','color','brand']
    for features in hav_null:
        data_set=data_set.loc[-data_set[features].isnull()]
        print("\n\nNo. of data points after eliminating {}=NULL :\n".format(features),data_set.shape[0])
    return data_set

#Near Duplicate items remain
#remove data differing only at end(that is size difference or colour difference are not good recommendations)
def RemoveDuplicateItemsAfterSorting(data_set):
    print("\nThe duplicated items are : ",sum(data_set.duplicated('title')),"\n\n")
    #sort on the basis of the title(alphabetical order)
    data_set.sort_values('title',inplace=True,ascending=False)
    #print(data_set['title'].head(20))
    
    row_wise_str_index=[]
    for row_index,row in data_set.iterrows():
        row_wise_str_index.append(row_index)
    new_data_list=[]
    i=0
    j=0
    len_data_points = data_set.shape[0]
    while i<len_data_points and j<len_data_points:
        temp_i=i
        #wordlist for ith string
        i_str = data_set['title'].loc[row_wise_str_index[i]].split()
        j=i+1
        while j<len_data_points:
            j_str = data_set['title'].loc[row_wise_str_index[j]].split()
            max_len=max(len(i_str),len(j_str))
            count=0 #no. of words matching in both strings
            #itertools.zip_longest(i_str,j_str) return correspoinding words in format [('a1','b1'),('a2','b2'),(None,'b3')]
            for words in itertools.zip_longest(i_str,j_str):
                if words[0] == words[1]:
                    count+=1
            if(max_len-count>2):
                #word difference>2 both different items -- hence include
                new_data_list.append(data_set['asin'].loc[row_wise_str_index[j]])
                if(max_len-1==j):
                    #word difference>2 but in len_data_points and len_data_points-1
                    new_data_list.append(data_set['asin'].loc[row_wise_str_index[j]])
                i=j
                break
            else:
                j+=1
        if temp_i == i:
            break   
    data_set=data_set.loc[data_set['asin'].isin(new_data_list)]
    return data_set

def RemoveDuplicateItemsNonAdjacent(data_set): # Time complexity = O(n^2) 
    row_wise_str_index=[]
    for row_index,row in data_set.iterrows():
        row_wise_str_index.append(row_index)
    new_data_list=[]
    while(len(row_wise_str_index)!=0):
        #remove the last element in the row_wise_str_index and return the index of the last element
        i = row_wise_str_index.pop()
        new_data_list.append(data_set['asin'].loc[i])
        i_str=data_set['title'].loc[i].split()
        for j in row_wise_str_index:
            j_str = data_set['title'].loc[j].split()
            max_len = max(len(i_str),len(j_str))
            count = 0 #count of mathing words in the two strings
            for words in itertools.zip_longest(i_str,j_str):
                if words[0] == words[1]:
                    count+=1
            # word difference > 3 the words are considered different 
            if (max_len-count) < 3:
                row_wise_str_index.remove(j)
    data_set=data_set.loc[data_set['asin'].isin(new_data_list)]
    return data_set
