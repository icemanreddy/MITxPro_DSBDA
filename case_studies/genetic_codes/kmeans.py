from numpy import loadtxt
from itertools import chain,product
import re
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import colorsys
'''
Input is an empty pandas dataframe(only headers populated) and a list of fragment strings.
Output is a pandas dataframe with the columns filled with frequency
'''
def calcFreq(df,fragments):
    #note the headers of the df
    headers=list(df)
    #add rows with 0 initialzied columns for each fragment
    df=pd.DataFrame([[0]*len(headers)]*len(fragments),columns=headers)

    #find out size of words (1,2,3 or 4)
    size=len(headers[0])
    #iterate over each fragment
    fragment_index=0
    for fragment in fragments:
        start=0
        end=0
        for start in range(0,len(fragment),size):
            end=start+size
            if end>len(fragment):
                #string is of insufficent length
                break
            sub_str=fragment[start:end]
            df.at[fragment_index,sub_str]=df.at[fragment_index,sub_str]+1
        fragment_index=fragment_index+1
    return df

inputfile=sys.argv[1]

string=''
R=re.compile('[^atgc]+')
#check if the input file is error free
with open(inputfile) as f:
    #line_num is the line in the file
    line_num=0
    if inputfile.endswith('.fa'):
        #skip comment
        line_num=line_num+1
        next(f)

    for line in f:
        line_num=line_num+1
        #strip newlines
        line=line.rstrip('\n')
        if R.search(line):
            print("invalid code at line:",str(line_num))
            print("Invalid code is :",R.search(line).group())
            exit()
        else:
            string=string+line
#split into fragments of 300 letters
fragments=[]

index=0
for index in range(0,len(string),300):
    start=index
    #index of last element is len(string)-1
    if start+300<len(string):
        end=start+300
    else:
        end=len(string)-1
    fragments.append(string[start:end])

num_tables=4
tables=[dict() for x in range(num_tables)]

#populate headers for the tables
gen_str=['a','t','c','g']
for i in range(num_tables):
    loc=0
    #https://docs.python.org/2/library/itertools.html#itertools.product
    for string in list(product(gen_str,repeat=i+1)):
        tables[i][''.join(string)]=[]
# populate an array of dataframes .Each dataframe whose rows are fragments and columns are 1 or 2 or 3 or 4 sized words
df=[]
X_norm=[]
fig=plt.figure(figsize=(20,10))
for i in range(num_tables):
    print("Initializing dataframe for ",i+1,"sized words.")
    df.append(pd.DataFrame(data=tables[i]))
    df[i]=calcFreq(df[i],fragments)
    #lets do PCA for words of various lengths
    X=np.array(df[i])
    print("Do PCA analysis for n=",i+1)
    pca=PCA(n_components=2)
    X_r=pca.fit_transform(X)
    X_norm.append(normalize(X_r,axis=0,norm='l1'))
    #print(normalize(X_r,axis=0,norm='l1'))
    ax=fig.add_subplot(2,2,i+1)
    ax.set_title(" PCA analysis with n="+str(i+1))
    ax.scatter(X_norm[i][:,0],X_norm[i][:,1],c="navy",s=2,alpha=0.8,lw=1)

plt.show()

#run k-means algo on words of length 3.
#Set K by roughly eyeballing the image
k=7
data=X_norm[2]
kmeans=KMeans(n_clusters=7,random_state=0).fit(data)
Y=kmeans.labels_
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(1,1,1)
ax.set_title(" K means clustering with K="+str(k))
#add centers to the data before showing
ax.scatter(data[:,0],data[:,1],c=Y,s=2)
X_c=kmeans.cluster_centers_[:,0]
Y_c=kmeans.cluster_centers_[:,1]
ax.scatter(X_c,Y_c,c=kmeans.predict(kmeans.cluster_centers_),s=70)
plt.show()
