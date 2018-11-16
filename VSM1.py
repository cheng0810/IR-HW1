import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

"""
cal all doc vocabulary 
"""
file_doc = open('C:/Users/Cheng/Desktop/jupyter/Homework1/doc_list.txt')        #

file_doc = file_doc.read().replace('\n',' ').split(' ')

doc_all=[]

for root in file_doc:
    doc = open('C:/Users/Cheng/Desktop/jupyter/Homework1/Document/' + root)
    # for i in range(3):
    #     doc.readline()
    doc.readline()          #use three line will faster than use for()
    doc.readline()
    doc.readline()
    doc = doc.read().replace(' -1\n',' ').split(' ')            #preprocessing
    del doc[-1]
    doc_all.extend(doc)
    
#print(len(doc_all))  #just check how many voc before delete repeat voc
doc_all = list(set(doc_all))            #delete repeat voc
#print(len(doc_all)) #check how many voc after delete

"""
cal all doc vocabulary 
"""
file_query = open('C:/Users/Cheng/Desktop/jupyter/Homework1/query_list.txt')

file_query = file_query.read().replace('\n',' ').split(' ')         #preprocessing

query_all=[]

for root in file_query:
    query = open('C:/Users/Cheng/Desktop/jupyter/Homework1/Query/' + root)
    query = query.read().replace(' -1\n',' ').split(' ')
    del query[-1]
    query_all.extend(query)
    
#print(len(query_all))  #check
query_all = list(set(query_all))
#print(len(query_all))  #check

"""
add both doc and query vocabulary 
"""
doc_all.extend(query_all)
#print(len(doc_all))   #check
doc_all = list(set(doc_all))
#print(len(doc_all))    #check

"""
cal the word appear how many times in each document
"""
doc_array = np.zeros((len(doc_all),2265 + 1))

for i in range(len(doc_all)):
    doc_array[i,0] = doc_all[i]

for i in range(len(file_doc)):       #有幾個doc檔案
    file0 = open('C:/Users/Cheng/Desktop/jupyter/Homework1/Document/' + file_doc[i])
    file0.readline()
    file0.readline()
    file0.readline()
    file0 = file0.read().replace(' -1\n',' ').split(' ')
    del file0[-1]
    for j in range(len(doc_all)):        #計算voc在各個doc出現次數
        if(file0.count(doc_all[j])) >0:
            doc_array[j,i+1] = 1 + math.log(file0.count(doc_all[j]),2)
        else:
            doc_array[j,i+1] = 0

# df_doc = pd.DataFrame(doc_array)
#
# df_doc       #轉成pandas比較好在jupyter上觀看

"""
cal the word appear how many times in each query
"""
query_array = np.zeros((len(doc_all),16 + 1))

for i in range(len(doc_all)):
    query_array[i,0] = doc_all[i]

# file = open('D:/downloads/Homework1/Query/20001.query')    #先測試一個檔案
# file = file.read().replace(' -1\n',' ').split(' ')

# for i in range(len(query_all)): #計算文本出現次數
#     if(file.count(query_all[i])) >0:
#         b[i,1] = 1 + math.log(file.count(query_all[i]),2)
#     else:
#         b[i,1] = 0
    
# df = pd.DataFrame(b)

# df

for i in range(len(file_query)): #有幾個query檔案
    file1 = open('C:/Users/Cheng/Desktop/jupyter/Homework1/Query/' + file_query[i])
    file1 = file1.read().replace(' -1\n',' ').split(' ')
    del file1[-1]
    for j in range(len(doc_all)): #計算voc在各個query出現次數
        if(file1.count(doc_all[j])) >0:
            query_array[j,i+1] = 1 + math.log(file1.count(doc_all[j]),2)
        else:
            query_array[j,i+1] = 0

# df_query = pd.DataFrame(query_array)
#
# df_query

"""
-----------------------finish the Term Fequency TF-------------------------------------
"""

doc_IDF = np.zeros((len(doc_all),2))

for i in range(len(doc_all)):
    doc_IDF[i,0] = doc_all[i]

for i in range(len(file_doc)):  #有幾個doc檔案
    file0 = open('C:/Users/Cheng/Desktop/jupyter/Homework1/Document/' + file_doc[i])
    file0.readline()
    file0.readline()
    file0.readline()
    file0 = file0.read().replace(' -1\n',' ').split(' ')
    del file0[-1]
    for j in range(len(doc_all)): #計算voc出現在多少個doc內
        if(file0.count(doc_all[j])) >0:
            doc_IDF[j,1] += 1

for i in range(len(doc_all)):
    if(doc_IDF[i,1]) >0:
        doc_IDF[i,1] = math.log(2265/doc_IDF[i,1],10)
    else:
        doc_IDF[i,1] = 0
    
# df_IDF = pd.DataFrame(doc_IDF)
#
# print(df_IDF)  #check in jupyter
"""
------------------------IDF finish---------------------------------
"""


"""
all_vocabulary TF-IDF
"""
for j in range(2265):
    for i in range(len(doc_all)):
        doc_array[i,j+1] = doc_array[i,j+1]*doc_IDF[i,1]


#doc_array  #check


"""
query TF-IDF
"""
for j in range(16):
    for i in range(len(doc_all)):
        query_array[i,j+1] = query_array[i,j+1]*doc_IDF[i,1]


#query_array    #check

"""
cal cos()
"""
doc_name = open('C:/Users/Cheng/Desktop/jupyter/Homework1/doc_list.txt')
doc_name = doc_name.read().replace('\n',' ').split(' ')

f = open('M10715090.txt','w')    #write in txt file
f.write('Query,RetrievedDocuments\n')
for k in range(16):         #from query 1 to 16
    result1 = np.zeros((2265,2),dtype=object)
    for i in range(2265):           #把cos()出來的值與doc名稱一一對應放入
        result1[i,1] = doc_name[i]
        result1[i,0] = cosine_similarity([query_array[:,k+1]],[doc_array[:,i+1]])

    p_result = pd.DataFrame(result1)            #use pandas to sort
    p_result = p_result.sort_values(ascending=False, by=[0]).values
        
    # for j in range(2264):           #用buble sort 比較容易理解
    #     for i in range(2264): #每回合進行比較的範圍
    #         if result1[i,0] < result1[i+1,0]: #是否需交換,若需要 則一併把doc名稱也一起交換
    #             tmp = result1[i,0]
    #             tmp2 = result1[i,1]
    #             result1[i,0] = result1[i+1,0]
    #             result1[i,1] = result1[i+1,1]
    #             result1[i+1,0] = tmp
    #             result1[i+1,1] = tmp2
    
    f.write(file_query[k])          #we have 16 querys file
    f.write(',')
    for i in range(2265):           #put every query result to txt
        f.write(p_result[i,1])
        f.write(' ')
        
    f.write('\n')
            

