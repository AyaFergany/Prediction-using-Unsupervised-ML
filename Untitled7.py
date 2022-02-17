#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Read Dataset
iris = pd.read_csv(r'C:\Users\zizo\Desktop\iris.csv')
iris


# In[7]:


#Drop unnecessary column
iris.drop('Id',axis=1,inplace = True)
iris


# In[8]:


sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',hue='Species',s=100,data=iris)
plt.title("ACTUAL'SepalLengthCm' vs 'PetalLengthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[9]:


sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',hue='Species',s=100,data=iris)
plt.title("ACTUAL'SepalWidthCm' vs 'PetalWidthCm'")
plt.xlabel('SepalWidthCm')
plt.ylabel('PetalWidthCm')
plt.show()


# In[10]:


x = iris.iloc[:, [0, 1, 2, 3]].values
model = KMeans(n_clusters=3)
model.fit(x)


# In[11]:


labels = model.predict(x)
labels


# In[12]:


xs = x[:,0]
ys = x[:,2]
plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.title("PREDICT'SepalLengthCm' vs 'PetalLengthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[13]:


xs = x[:,1]
ys = x[:,3]
plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.title("PREDICT'SepalWidthCm' vs 'PetalWidthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[14]:


#centroid
centroids = model.cluster_centers_


# In[15]:


centroids_x = centroids[:,0]
centroids_y = centroids[:,2]
plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.scatter(centroids_x,centroids_y,s=100,marker='D',alpha=0.9,c='green')
plt.title("PREDICT'SepalLengthCm' vs 'PetalLengthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[16]:


centroids_x = centroids[:,1]
centroids_y = centroids[:,3]
plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.scatter(centroids_x,centroids_y,s=100,marker='D',alpha=0.9,c='green')
plt.title("PREDICT'SepalWidthCm' vs 'PetalWidthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[17]:


#ACTUAL VS PREDICT
plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.title("PREDICT'SepalLengthCm' vs 'PetalLengthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()

sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',hue='Species',s=100,data=iris)
plt.title("ACTUAL'SepalLengthCm' vs 'PetalLengthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()


# In[18]:


plt.scatter(xs, ys,c=labels,alpha=0.9)
plt.title("PREDICT'SepalWidthCm' vs 'PetalWidthCm'")
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()

sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',hue='Species',s=100,data=iris)
plt.title("ACTUAL'SepalWidthCm' vs 'PetalWidthCm'")
plt.xlabel('SepalWidthCm')
plt.ylabel('PetalWidthCm')
plt.show()


# In[19]:


i=iris['Species']
print(i)


# In[20]:


df=pd.DataFrame({'labels':labels , 'species':i})
print(df)


# In[21]:


#DO THE CLUSTER CORRESPOND TO THE SPECIES???
#- we see that cluster 1 corresponds perfectly with the species setosa
#- on the other hand ,while cluster 0 contains mainly virginica samples,there are also some virginica in cluster 2
ct=pd.crosstab(df['labels'],df['species'])
print(ct)


# In[22]:


#Measuring clustering quality
#Elbow Method
#lower value of inertia is better
# k-means attempts to minimize the intertia when choosing clusters
print(model.inertia_)


# In[23]:


#clusterings of the iris dataset with different numbers of clusters
#our k-means model with 3 cluster has relativety low inertia ,which is great ,
#for iris dataset,3 is a good choice
ks = range(1, 7)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    
    # Fit model to samples
    model.fit(x)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.title("Elbow Method")
plt.xticks(ks)
plt.show()


# In[ ]:




