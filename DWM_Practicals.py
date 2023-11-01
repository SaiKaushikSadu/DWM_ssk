#!/usr/bin/env python
# coding: utf-8

# # K Means

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


#function for euclidean distance
def eu_distance(p1,p2):
    return np.sqrt(np.sum(p1-p2)**2)


# In[14]:


#initialize cluster
def initialize(data,k):
    np.random.shuffle(data)
    return data[:k]


# In[33]:


data = np.array([[1, 2], [2, 3], [8, 7], [7, 6], [1, 1], [8, 8]])
k=2
centroids=initialize(data,k)
print(centroids)


# In[34]:


#assign to cluster
def assign(data,centroids):
    clusters=[[] for _ in range(len(centroids))]
    for point in data:
        distance=[eu_distance(point,centroid) for centroid in centroids]
        cluster_index=np.argmin(distance)
        clusters[cluster_index].append(point)
    return clusters


# In[35]:


clusters=assign(data,centroids)
print(clusters)


# In[36]:


#update centroids
def update(clusters):
    new_centroids=[np.mean(cluster,axis=0) if cluster else np.nan for cluster in clusters]
    return np.array(new_centroids)


# In[37]:


up=update(clusters)
print(up)


# In[38]:


def converge(old_centroids,new_centroids,tol=1e-4):
    return np.all(np.abs(old_centroids-new_centroids)<tol)


# In[39]:


def k_means(data,k,max_it=100):
    centroids=initialize(data,k)
    for i in range(max_it):
        clusters=assign(data,centroids)
        new_centroids=update(clusters)
        if converge(centroids,new_centroids):
            break
        centroids=new_centroids
    return centroids, clusters


# In[40]:


final_centroids,final_clusters=k_means(data,2)
print(final_centroids)
print(final_clusters)


# In[43]:


for i, cluster in enumerate(final_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('K-means Clustering')
plt.show()


# # Apriori Algorithm

# In[1]:


data = [
        ['T100',['I1','I2','I5']],
        ['T200',['I2','I4']],
        ['T300',['I2','I3']],
        ['T400',['I1','I2','I4']],
        ['T500',['I1','I3']],
        ['T600',['I2','I3']],
        ['T700',['I1','I3']],
        ['T800',['I1','I2','I3','I5']],
        ['T900',['I1','I2','I3']]
        ]

#sorting
init = []
for i in data:
    for q in i[1]:
        if(q not in init):
            init.append(q)
init = sorted(init)
print(init)

#support
sp = 0.4
s = int(sp*len(init))
s

from collections import Counter

#counter for c
c = Counter()
for i in init:
    for d in data:
        if(i in d[1]):
            c[i]+=1
print("C1:")
for i in c:
    print(str([i])+": "+str(c[i]))
print()

# counter for l
l = Counter()
for i in c:
    if(c[i] >= s):
        l[frozenset([i])]+=c[i]
print("L1:")
for i in l:
    print(str(list(i))+": "+str(l[i]))
print()


pl = l
pos = 1
for count in range (2,5):
    nc = set()
    temp = list(l)
#     print(temp)
    for i in range(0,len(temp)):
        for j in range(i+1,len(temp)):
            t = temp[i].union(temp[j])
            if(len(t) == count):
                nc.add(temp[i].union(temp[j]))
    nc = list(nc)

    c = Counter()
    for i in nc:
        c[i] = 0
        for q in data:
            temp = set(q[1])
            if(i.issubset(temp)):
                c[i]+=1
    print("C"+str(count)+":")
    for i in c:
        print(str(list(i))+": "+str(c[i]))
    print()

    l = Counter()
    for i in c:
        if(c[i] >= s):
            l[i]+=c[i]
    print("L"+str(count)+":")
    for i in l:
        print(str(list(i))+": "+str(l[i]))
    print()
    if(len(l) == 0):
        break
    pl = l
    pos = count
print("Result: ")
print("L"+str(pos)+":")
for i in pl:
    print(str(list(i))+": "+str(pl[i]))
print()

confidence=50

from itertools import combinations
for l in pl:
    c = [frozenset(q) for q in combinations(l,len(l)-1)]
    mmax = 0
    for a in c:
        b = l-a
#         print("lenene",b)
        ab = l
        sab = 0
        sa = 0
        sb = 0
        for q in data:
            temp = set(q[1])
            if(a.issubset(temp)):
                sa+=1
            if(b.issubset(temp)):
                sb+=1
            if(ab.issubset(temp)):
                sab+=1
        temp = sab/sa*100
        if(temp >= confidence):
            mmax = temp
        temp = sab/sb*100
        if(temp >= confidence):
            mmax = temp
        print(str(list(a))+" -> "+str(list(b))+" = "+str(sab/sa*100)+"%")
        print(str(list(b))+" -> "+str(list(a))+" = "+str(sab/sb*100)+"%")
    curr = 1
    print("choosing:", end=' ')
    for a in c:
        b = l-a
        ab = l
        sab = 0
        sa = 0
        sb = 0
        for q in data:
            temp = set(q[1])
            if(a.issubset(temp)):
                sa+=1
            if(b.issubset(temp)):
                sb+=1
            if(ab.issubset(temp)):
                sab+=1
        temp = sab/sa*100
        if(temp >= confidence):
            print(curr, end = ' ')
        curr += 1
        temp = sab/sb*100
        if(temp >= confidence):
            print(curr, end = ' ')
        curr += 1
    print()
    print()


# # Naive Bayes

# In[19]:


import pandas as pd
import math

data = [
    {'Color': 'R', 'Type': "S", 'Origin': 'D', 'Stolen': 'Yes'},
    {'Color': 'R', 'Type': "S", 'Origin': 'D', 'Stolen': 'No'},
    {'Color': 'R', 'Type': "S", 'Origin': 'D', 'Stolen': 'Yes'},
    {'Color': 'Y', 'Type': "S", 'Origin': 'D', 'Stolen': 'No'},
    {'Color': 'Y', 'Type': "S", 'Origin': 'I', 'Stolen': 'Yes'},
    {'Color': 'Y', 'Type': "U", 'Origin': 'I', 'Stolen': 'No'},
    {'Color': 'Y', 'Type': "U", 'Origin': 'I', 'Stolen': 'Yes'},
    {'Color': 'Y', 'Type': "U", 'Origin': 'D', 'Stolen': 'No'},
    {'Color': 'R', 'Type': "U", 'Origin': 'I', 'Stolen': 'No'},
    {'Color': 'R', 'Type': "S", 'Origin': 'I', 'Stolen': 'Yes'}
]

color = input("Enter 'Color' (R,Y): ")
type = (input("Enter 'Type' (S,U): "))
origin = input("Enter 'Origin' (D,I): ")

print("\n")

# Yes and No ka khel

# Prior probabilities
total_instances = len(data)

stolen_yes=0
stolen_no=0
for d in data:
    if(d['Stolen']=="Yes"):
        stolen_yes+=1

for d in data:
    if(d['Stolen']=="No"):
        stolen_no+=1
        
p_yes = round(stolen_yes / total_instances, 4)
p_no = round(stolen_no / total_instances, 4)

print(f"Total occurrences of Yes: {stolen_yes}")
print(f"Total occurrences of No: {stolen_no}")
print(f"Prior probability for Yes: {p_yes}")
print(f"Prior probability for No: {p_no}")

# Attributes ka khel

# Likelihood probabilities
def calculate_likelihood(attribute, value, stolen):
    subset = [d for d in data if d['Stolen'] == stolen]
    count = len([d for d in subset if d[attribute] == value])
    total = len(subset)
    return count / total

print("\n")
print("\n")
print("The likelihood probabilities are: ")

p_color_given_yes = round(calculate_likelihood('Color', color, 'Yes'), 4)
p_color_given_no = round(calculate_likelihood('Color', color, 'No'), 4)

print(f"P(color = {color} / buy='Yes') = {p_color_given_yes}")
print(f"P(color = {color} / buy='No') = {p_color_given_no}")
print("\n")

p_type_given_yes = round(calculate_likelihood('Type', type, 'Yes'), 4)
p_type_given_no = round(calculate_likelihood('Type', type, 'No'), 4)

print(f"P(type = {type} / buy='Yes') = {p_type_given_yes}")
print(f"P(type = {type} / buy='No') = {p_type_given_no}")
print("\n")

p_origin_given_yes = round(calculate_likelihood('Origin', origin, 'Yes'), 4)
p_origin_given_no = round(calculate_likelihood('Origin', origin, 'No'), 4)

print(f"P(origin = {origin} / buy='Yes') = {p_origin_given_yes}")
print(f"P(origin = {origin} / buy='No') = {p_origin_given_no}")
print("\n")

# Posterior probabilities
print("The posterior probabilities are: ")

p_yes_given_x = round(p_yes * p_color_given_yes * p_type_given_yes * p_origin_given_yes , 4)
p_no_given_x = round(p_no * p_color_given_no * p_type_given_no * p_origin_given_no , 4)

if p_yes_given_x > p_no_given_x:
    prediction = 'Yes'
else:
    prediction = 'No'

print(f'Probability of stolen=Yes: {p_yes_given_x}')
print(f'Probability of stolen=No: {p_no_given_x}')
print(f'Prediction: {prediction}')


# # Linear Regression

# In[4]:


x = [45, 70, 60, 84, 75, 84]
y = [60, 75, 54, 82, 68, 76]

mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
deno = sum((xi - mean_x) ** 2 for xi in x)
b = num / deno
a = mean_y - b * mean_x

print(f"Coefficient a: {a}")
print(f"Coefficient b: {b}")

print(f"The equation of the line is: y = {a} + {b}x")


# # Page Rank

# In[11]:


import numpy as np

def calculate_pagerank(adjacency_matrix, damping_factor=0.85, max_iterations=100, tolerance=1e-5):
    num_pages = len(adjacency_matrix)
    teleport_prob = (1 - damping_factor) / num_pages
    pagerank = np.full(num_pages, 1.0 / num_pages)

    for _ in range(max_iterations):
        new_pagerank = np.zeros(num_pages)
        for i in range(num_pages):
            for j in range(num_pages):
                if adjacency_matrix[j][i] == 1:
                    num_links_on_page_j = sum(adjacency_matrix[j])
                    new_pagerank[i] += damping_factor * (pagerank[j] / num_links_on_page_j)
            new_pagerank[i] += teleport_prob
            
        # Check for convergence
        if np.sum(np.abs(new_pagerank - pagerank)) < tolerance:
            return new_pagerank
        pagerank = new_pagerank
    return pagerank

n = int(input('Enter the total number of vertex:'))
print('Enter the transition matrix:')
graph = [[] for _ in range(n)]

# input adjacency
for i in range(n):
    graph[i] = [int(val) for val in input().split()]


pagerank_scores = calculate_pagerank(graph)

wholes=[(i,pagerank_scores[i]) for i in range(len(graph))]
wholes=sorted(wholes,key=lambda x:x[1],reverse=True)
cnt=1
for (i, score) in wholes:
    print(f'Rank {cnt} --> Page {i + 1}: {score:.4f}')
    cnt+=1


# # Heriarchial Clustering - Algomeritive

# In[18]:


import pandas as pd
import math
import scipy.cluster.hierarchy as shc  
import matplotlib.pyplot as plt

# data = pd.read_csv('./heirch.csv')
# cod = data[['X', 'Y']].values
cod=[[0.40,0.53],[0.22,0.38],[0.35,0.32],[0.26,0.19],[0.08,0.41],[0.45,0.30]]

distance = [[math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for x1, y1 in cod] for x2, y2 in cod]
df = pd.DataFrame(distance)

clusters = {i: [i] for i in range(len(cod))}
print(clusters)

def agglomerate(dist):
    min_distance = float('inf')
    min_indices = [0, 0]

    for i in range(len(dist)):
        for j in range(i):
            if min_distance > dist.iloc[i, j]:
                min_distance = dist.iloc[i, j]
                min_indices = [dist.columns[j], dist.index[i]]

    min_indices.sort()
    print(f'[P{clusters[min_indices[0]]}, P{clusters[min_indices[1]]}]')
    print("Minimum distance:", min_distance, "\n\n")
    dist = dist.drop(index=min_indices[1], columns=min_indices[1])

    for i in range(len(dist)):
        for j in range(len(dist)):
            curr = min([distance[m][n] for m in clusters[dist.columns[j]] for n in clusters[dist.index[i]]])
            dist.iloc[i, j] = curr

    clusters[min_indices[0]].extend(clusters[min_indices[1]])
    del clusters[min_indices[1]]

    return dist


while len(df) > 1:
    df = agglomerate(df)


# Now, create the dendrogram manually
# plt.figure(figsize=(10, 5))
# plt.xlabel("Sample Index")
# plt.ylabel("Distance")
# plt.title("Dendrogram")

# dendrogram_data = []
# for i in range(len(cod)):
#     dendrogram_data.append([i, i, 0, 0])  # Format: [left, right, distance, number of points in cluster]

# for i, (left, right, _, num_points) in enumerate(dendrogram_data):
#     x = [left, right]
#     y = [0, dendrogram_data[i - 1][2] if i > 0 else 0]
#     plt.plot(x, y, color='b')

# labels = [f'P{i + 1}' for i in range(len(cod))]
# plt.xticks(range(len(labels)), labels)
# plt.show()

dendro = shc.dendrogram(shc.linkage(cod, method="ward"))  
plt.title("Dendrogrma Plot")  
plt.ylabel("Euclidean Distances")  
plt.xlabel("Customers")  
plt.show()  

