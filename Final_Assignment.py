# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:50:31 2021

@author: Amartya
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Reading of Document File
books=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\Recommendation Engine/book.csv",encoding='unicode_escape')
books.Rating.value_counts()
books['Rating']=books['Rating'].map({1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine',10:'Ten'})

#Calculating Cosine Similarity Matrix
tfidf=TfidfVectorizer()
mat=tfidf.fit_transform(books.Rating)   
cos_sm=linear_kernel(mat,mat)
b_ind=pd.Series(books.index,index=books.Title).drop_duplicates()

#Recommendation Engine Function
def recomend(Name,topN):
    b_id=b_ind[Name]
    cos_sc=list(enumerate(cos_sm[b_id]))
    cos_sc=sorted(cos_sc,key= lambda x:x[1],reverse=True)
    cos_scN=cos_sc[0:topN+1]
    book=[i[0] for i in cos_scN]
    score=[i[1] for i in cos_scN]
    similar=pd.DataFrame(columns=["Book_Name","Scores"])
    similar["Book_Name"]=books.loc[book,"Title"]
    similar["Scores"]=score
    similar.reset_index(inplace=True)
    similar.drop(["index"],axis=1,inplace=True)
    print(similar)

    
recomend("Classical Mythology",15)

