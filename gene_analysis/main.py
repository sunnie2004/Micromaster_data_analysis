# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

#####################
# #problem1
X = np.load('F:/mit-micromaster/machinelearning/gene_analysis/data/p1/X.npy', mmap_mode='r')  ##get the file X-
# # # # print(X.shape)
# # #
# # # # print(np.max(X[:,0]))
X_log = np.log2(X+1)
print(np.max(X_log[:,0])) #transform X

pca_log = PCA(n_components=2)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca_log.fit(X_log)
principalComponents = pca_log.fit_transform(X_log)
# # print("PCA:",principalComponents)
# # print(pca_log.components_)
# # print((pca_log.components_).shape)# print pca_log.components_
# # print("varaince ratio after log:",pca_log.explained_variance_ratio_)
# # print ("maximum process data:",np.max(pca_log.components_[:,0]))
# # print (pca_log.components_).shape
# # print((pca_log.explained_variance_ratio_).shape)
sum_log = np.cumsum(pca_log.explained_variance_ratio_)
# # print np.where(sum_log >= 0.85)
#
pca = PCA()   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca.fit(X)
# print("varaince ratio raw:",pca.explained_variance_ratio_)
sum_raw = np.cumsum(pca.explained_variance_ratio_)
# print np.where(sum_raw >= 0.85)

# Y = np.load('/Users/JohnCook/Downloads/data/p1/Y.npy', mmap_mode='r')  #file Y -ground truth label
# print Y
x = principalComponents[:,0]
y = principalComponents[:,1]
# area = np.pi*3
colors = (0,0,0)

plt.scatter(x, y, c=colors, alpha=0.5)
plt.title('Scatter plot PCA')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

##########
#MDS

embedding = MDS(n_components=2)
MDSComponents = embedding.fit_transform(X_log)
print("MDS:",MDSComponents)

x = MDSComponents[:,0]
y = MDSComponents[:,1]
# area = np.pi*3
colors = (0,0,0)

plt.scatter(x, y, c=colors, alpha=0.5)
plt.title('Scatter plot MDS')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#####################################
#TSNE

pca_50 = PCA(n_components=50)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca_result_50 = pca_50.fit_transform(X_log)

i=1

fig=plt.figure()
for component in [10,50,100,250,500]:
    pca_50 = PCA(n_components=component)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
    pca_result_50 = pca_50.fit_transform(X_log)
    embedding = TSNE(n_components=2,perplexity=40)
    TSNEComponents = embedding.fit_transform(pca_result_50)


    x = TSNEComponents[:,0]
    y = TSNEComponents[:,1]
    ax = fig.add_subplot(510+i)  #straight,herison, number of current
    i+=1
    ax.scatter(x, y,c='r')
    ax.set_title('Scatter plot TSNE with  number of PC ' + str(component))

plt.show()

for component in [10,50,100,250,500]:
    pca_50 = PCA(n_components=component)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
    pca_result_50 = pca_50.fit_transform(X_log)
    embedding = TSNE(n_components=2,perplexity=40)
    TSNEComponents = embedding.fit_transform(pca_result_50)


    x = TSNEComponents[:,0]
    y = TSNEComponents[:,1]

    plt.scatter(x, y,c='b')
    plt.title('Scatter plot TSNE with  number of PC ' + str(component))
    plt.show()
#############################################
##learning_rate
for learning_rates in [0.001,10,200,1000,2000]:
    pca_50 = PCA(n_components=50)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
    pca_result_50 = pca_50.fit_transform(X_log)
    embedding = TSNE(n_components=2,perplexity=40,learning_rate=learning_rates)
    TSNEComponents = embedding.fit_transform(pca_result_50)


    x = TSNEComponents[:,0]
    y = TSNEComponents[:,1]

    plt.scatter(x, y,c='b')
    plt.title('Scatter plot TSNE with  learning_rate ' + str(learning_rates))
    plt.show()
#################################################
#k-mean
#############################
# #elbow method
wcss = []
for i in range(1, 11):
   kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
   kmeans.fit(pca_result_50)
   #appending the WCSS to the list (kmeans.inertia_ returns the WCSS value for an initialized cluster)
   wcss.append(kmeans.inertia_)
   print(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

n_clusters = 5
kmean = KMeans(n_clusters=n_clusters)
kmean.fit(pca_result_50)
print("kmean: k={}, cost={}".format(n_clusters, float(kmean.score(pca_result_50))))
#############################
#k_mean before PCA/MDS/TSNE
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmean_50 = kmeans.fit_transform(pca_result_50)
y = kmeans.fit_predict(pca_result_50)   #color


# ############
#PCA
# embedding_pca = PCA()
# PCAComponents_50 = embedding_pca.fit_transform(kmean_50)
# plt.scatter(PCAComponents_50[:, 0], PCAComponents_50[:, 1],c=y)
# plt.show()

############
#MDS
embedding = MDS(n_components=2)
MDSComponents_50 = embedding.fit_transform(kmean_50)
plt.scatter(MDSComponents_50[:, 0], MDSComponents_50[:, 1],c=y)
plt.show()

############
#TSNE
i=0
fig=plt.figure()
# embedding = TSNE(n_components=2,perplexity=40)
# TSNEComponents_50 = embedding.fit_transform(kmean_50)
# plt.scatter(TSNEComponents_50[:, 0], TSNEComponents_50[:, 1],c=y)
# plt.show()
for perplexity in [2,5,30,40,80,100]:
    embedding = TSNE(n_components=2,perplexity=perplexity)
    TSNEComponents = embedding.fit_transform(kmean_50)
    # print("TSNE:",TSNEComponents)
    ax =  fig.add_subplot(321+i)
    i+=1
    ax.scatter(TSNEComponents[:,0],TSNEComponents[:,1], c=y)
    ax.set_title('Scatter plot TSNE with perplexity '+str(perplexity))

plt.show()
###############
#computation mean
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmean_original = kmeans.fit_transform(X)

labels = kmeans.fit(X).labels_
centers = kmeans.fit(X).cluster_centers_
y = kmeans.fit_predict(centers)
print("k-mean of original data:",centers.shape)

# ############
#PCA
embedding_pca = PCA()
PCAComponents_original = embedding_pca.fit_transform(centers)
plt.scatter(PCAComponents_original[:, 0], PCAComponents_original[:, 1],c=y)
plt.show()

############
#MDS
embedding = MDS(n_components=2)
MDSComponents_original = embedding.fit_transform(centers)
plt.scatter(MDSComponents_original[:, 0], MDSComponents_original[:, 1],c=y)
plt.show()

#TSNE
embedding = TSNE(n_components=2,perplexity=40)
TSNEComponents_original = embedding.fit_transform(centers)
plt.scatter(TSNEComponents_original[:, 0], TSNEComponents_original[:, 1],c=y)
plt.show()
#########################################################
#problem2
###########################################################
#X_2 = np.load('F:/mit-micromaster/machinelearning/data/p2_unsupervised/X.npy', mmap_mode='r')  ##get the file X-
X_2 = np.load('F:/mit-micromaster/machinelearning/gene_analysis/data/p2_unsupervised_reduced/X.npy', mmap_mode='r')  ##get the reduced size file X-
print(X_2)
X2_log = np.log2(X_2+1)  #log transform

# print("log of original data:",X2_log)
print("dimension of original data:",X2_log.shape)
#
pca = PCA(n_components=50)   #Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca_result = pca.fit_transform(X2_log)  #PCA
####################
# #Hierarchical Clustering Dendrogram
import pandas as pd
Z = hierarchy.linkage(pca_result, 'ward')  #ward,single,complete,average,weighted,centroid
hierarchy.set_link_color_palette(['blue','green', 'red'])
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('subtype of samples')
plt.ylabel('distance')
labelList = ['cluster2_1','cluster2_2','cluster2_3','cluster3_1','cluster3_2','cluster3_3','cluster1_1','cluster1_2','cluster1_3']
# labelList = ['cluster1','cluster2','cluster3']
R = hierarchy.dendrogram(
                Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=9,  # show only the last p merged clusters
                no_plot=True,
                )
##leaf_label_func call function
temp = {R["leaves"][ii]: labelList[ii] for ii in range(len(R["leaves"]))}

def llf(xx):
    return temp[xx]
hierarchy.dendrogram(
    Z,
    color_threshold=1500,
    above_threshold_color='grey',
    truncate_mode='lastp',  #lastp
  #  p=9,
    leaf_rotation=90., # rotates the x axis labels
    leaf_font_size=12., # font size for the x axis labels
    show_contracted=True
 #   leaf_label_func=llf
)
# Add horizontal line.
plt.axhline(y=1500, c='grey', lw=1, linestyle='dashed')
plt.show()
# ####################
# #similar with kmeans,  but do not need give the number of clusters ahead, visualization shows 3 big clusters.
# #####
i=0
fig = plt.figure(figsize=(15, 10))
for linkage_type in ['single','complete','average','ward']:
    cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage=linkage_type)
    cluster.fit_predict(pca_result)

    ax = fig.add_subplot(221+i)
    i+=1
    ax.scatter(pca_result[:,0], pca_result[:,1], c=cluster.labels_)
    ax.set_title('hierarchical clustering type '+str(linkage_type),size=18)
plt.show()
#
# # #################################
# # #tsne  get the proper perplexity
# for perplexity in[5,10,50,40,80,100]:
#     embedding = TSNE(n_components=2,init='random', random_state=0,perplexity=perplexity)
#     TSNEComponents = embedding.fit_transform(pca_result)
#
#     x = TSNEComponents[:,0]
#     y = TSNEComponents[:,1]
#
#     plt.scatter(x, y)
#     plt.title('Scatter plot TSNE '+str(perplexity),size=18)
#     plt.xlabel('x',size=14)
#     plt.ylabel('y',size=14)
#     plt.show()
# #################
# # #kmean--sum of squares, inertia_  12
# # elbow method to find the best cluster number
cluster_sequence = 12
wcss = []
for i in range(1,cluster_sequence):
    all_mean = KMeans(i, n_init=10, max_iter=300, random_state=0)
    all_mean.fit(pca_result)
    wcss.append(all_mean.inertia_)
plt.plot(np.arange(1,cluster_sequence),wcss)
plt.title("Elbow Method for Optimal K",size = 18)
plt.xlabel("# clusters",size = 14)
plt.ylabel("within-Cluster Sum of Squares Distance", size = 14)
plt.show()
#########################done
##################
#silhouette to find the proper cluster number
from sklearn.metrics import silhouette_score,silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.style.colors import resolve_colors
##############
# #silhouette for optimal cluster
scores = [0]
for i in range(2, 12):
    fitx = KMeans(n_clusters=i, init='random', n_init=30, random_state=0).fit(X2_log)
    score = silhouette_score(X2_log, fitx.labels_)
    scores.append(score)

plt.figure(figsize=(12, 8.5))
plt.plot(range(1, 12), np.array(scores), 'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Average Silhouette')
plt.title('Average Silhouette Score showing the optimal $k$')
plt.show()
######################it works
silhouette_score_values = [0]
NumberOfClusters = range(2, 15)
for i in NumberOfClusters:
    classifier = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=109).fit(X2_log)

    score = silhouette_score(X2_log, classifier.labels_)
    silhouette_score_values.append(score)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.plot(np.arange(1,15), np.array(silhouette_score_values))
    plt.title("Silhouette score values vs Numbers of Clusters ")
plt.show()

#############
# #four figures for silhouette
fig, ax = plt.subplots(2, 2, figsize=(15,8))
location=np.array([[0,0],[0,1],[1,0],[1,1]])
k=0
for i in [2, 3, 4, 7]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)

    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors="yellowbrick",ax=ax[location[k][0]][location[k][1]],title="Silhouettte score Cluster #"+str(i))
    k+=1
    visualizer.fit(X2_log)
    visualizer.show()

plt.plot(np.arange(2,20),[silhouette_score(X2_log,KMeans(i,n_init=10,max_iter=300,random_state=0).fit(X2_log).predict(X2_log)) for i in range(2,20)])

###############################
#kmeans  find 3 clusters and sub-clusters good
kmeans = KMeans(n_clusters = 7, init = 'k-means++', n_init=10,random_state = 0)
kmean_pca = kmeans.fit_transform(pca_result)
label = kmeans.labels_

centers = kmeans.cluster_centers_
color = kmeans.fit_predict(pca_result)   #color
labels = kmeans.labels_

plt.scatter(pca_result[color==0, 0], pca_result[color==0, 1], s=80, label ='subtype 1')
plt.scatter(pca_result[color==1, 0], pca_result[color==1, 1], s=80, label ='subtype 2')
plt.scatter(pca_result[color==2, 0], pca_result[color==2, 1], s=80, label ='subtype 3')
plt.scatter(pca_result[color==3, 0], pca_result[color==3, 1], s=80, label ='subtype 4')
plt.scatter(pca_result[color==4, 0], pca_result[color==4, 1], s=80, label ='subtype 5')
plt.scatter(pca_result[color==5, 0], pca_result[color==5, 1], s=80, label ='subtype 6')
plt.scatter(pca_result[color==6, 0], pca_result[color==6, 1], s=80, label ='subtype 7')

plt.scatter(centers[:,0],centers[:,1],s=250, marker='*',edgecolor='black',c=np.arange(0,7))
plt.title('Scatter plot kmeans',size=18)
plt.xlabel('x',size=14)
plt.ylabel('y',size=14)
plt.legend(loc="lower right")
plt.show()
########################

#######################################
#kmeans-tsne
# kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
# kmean_pca = kmeans.fit_transform(pca_result)
# centers = kmeans.cluster_centers_
# color = kmeans.fit_predict(pca_result)   #color
# embedding = TSNE(n_components=2,perplexity=50)
# TSNEComponents = embedding.fit_transform(pca_result)
# #
# x = TSNEComponents[:,0]
# y = TSNEComponents[:,1]
# #
# plt.scatter(x, y, c=color)
# # plt.scatter(centers[:,0],centers[:,1], marker='*', label='centroids',edgecolor='black',c=np.arange(0,3))
# plt.title('Scatter plot kmeans-TSNE',size=18)
# plt.xlabel('x',size=14)
# plt.ylabel('y',size=14)
# plt.show()

######################
#change perplexity to test ????????more explanation before upload
# for perplexity in [2,5,30,50,100]:
#     kmeans = KMeans(n_clusters = 3,n_init=50, init = 'k-means++', random_state = 0)
#     kmeans.fit(pca_result)
#     color = np.array(resolve_colors(3,'yellowbrick'))
#
#     embedding = TSNE(n_components=2,perplexity=perplexity)
#     TSNEComponents = embedding.fit_transform(pca_result)
#
#     x = TSNEComponents[:,0]
#     y = TSNEComponents[:,1]
#
#     plt.scatter(x, y, c=color[kmeans.labels_])
#     plt.title('Scatter plot TSNE perplexity '+str(perplexity),size=18)
#     plt.xlabel('x',size=14)
#     plt.ylabel('y',size=14)
#     plt.show()
##########################################

#####################
#logistic regression
#find the best parameters
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init=10,random_state = 0).fit(X2_log)
labels = kmeans.labels_    #label the data according to the cluster number: 3

perm = np.random.permutation(X2_log.shape[0]) #shuffle data

n_train = int(7/10*X2_log.shape[0])
X2_train = X2_log[perm[:n_train]]
Y_train = labels[perm[:n_train]]
X2_test = X2_log[perm[n_train:]]
Y_test = labels[perm[n_train:]]
log_reg =[]
penalties =['l1','l2'] #'elasticnet': both L1 and L2 penalty terms are added.

for index,penalty in enumerate(penalties):  #cross-validation
    log_reg.append(LogisticRegressionCV(cv=5,Cs=[0.001,0.01,0.1,1,10],max_iter=3000,penalty=penalty,solver='liblinear',
                                        multi_class='ovr'))
    log_reg[index].fit(X2_train,Y_train)
    print("penalty:",penalty)
    print("Training Score:",log_reg[index].score(X2_train,Y_train))
    print("Test Score:",log_reg[index].score(X2_test,Y_test))
    print("Coef:",log_reg[index].coef_)
    print("C_:",log_reg[index].C_)
    #### check the performance of the model
    Y_test_pred = log_reg[index].predict(X2_test)
    rmse = np.sqrt(mean_squared_error(Y_test,Y_test_pred))
    print("Mean squared error:",rmse)
    accuracy_test = accuracy_score(Y_test,Y_test_pred)
    R2_Sore_test = r2_score(Y_test,Y_test_pred )
    print("Accuracy test data: %.2f%%:" %(accuracy_test*100))
    print("R2_score calculated of test data:", R2_Sore_test)
######################################
# result:penalty: l1
# Training Score: 1.0
# Test Score: 0.9493087557603687
# Coef: [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# C_: [ 0.1  0.1  1.   0.1 10.  10.  10.   0.1  1. ]
# Mean squared error: 0.43467239636983923
# Accuracy test data: 94.93%:
# R2_score calculated of test data: 0.9715071704445739
# penalty: l2
# Training Score: 1.0
# Test Score: 0.9692780337941628
# Coef: [[-1.55341873e-03 -4.49127962e-05 -9.40593069e-05 ...  2.23756504e-04
#   -2.45154704e-04 -2.55061036e-04]
#  [ 4.68376930e-03 -1.89566649e-04 -1.93504253e-04 ...  1.64756055e-03
#    2.84023456e-04  1.45367010e-03]
#  [ 2.01499617e-03 -2.55616548e-04 -6.46952012e-05 ... -1.66294442e-03
#    4.04662655e-04 -3.07150519e-03]
#  ...
#  [-1.58760820e-03 -4.84470690e-05 -3.34971805e-04 ...  1.37843149e-03
#    2.23618925e-03  1.64201587e-03]
#  [-1.05792376e-03 -7.00912916e-06 -2.20769078e-04 ... -7.34626188e-04
#   -9.44990682e-04 -9.15918180e-04]
#  [-1.09809651e-03 -1.04845900e-04 -2.21249284e-04 ...  3.48213861e-04
#   -4.08499313e-05 -7.82368004e-04]]
# C_: [1.e-03 1.e+00 1.e+01 1.e-03 1.e+00 1.e+01 1.e-01 1.e-03 1.e-02]
# Mean squared error: 0.39969266226733924
# Accuracy test data: 96.93%:
# R2_score calculated of test data: 0.9759085018393145
# conclusion: choose L2

#################################################################################################
#I did not figure out this way, split the data into train and test data，
# parameters = {'C':[0.001, 0.01,0.1,10, 100, 1000],
#          'penalty' :['l1','l2','elasticnet'],
#             'solver':['saga'],'max_iter':[1000]
# }
# lr = LogisticRegression()
# clf = GridSearchCV(estimator=lr, param_grid=parameters,n_jobs=-1,cv = 5)
# clf.fit(X2_train,Y_train)
# print('best accuracy:'% clf.best_score_)
# best_parameters = clf.best_estimator_.get_params()
# print('best parameters:',best_parameters)
# Y_test_pred = clf.predict(X2_test)
# mse = np.sqrt(mean_squared_error(Y_test,Y_test_pred))
# print("Mean squared error:",mse)
# size = np.arange(len(Y_test))
# plt.plot(size,sorted(Y_test_pred),c='black',linewidth=2,label='Predict')
# plt.plot(size,sorted(Y_test),c='black',linewidth=2,label='Test')
# plt.legend()
# plt.show()
################################################################

##########################
#problem2 part2.3
import random
#select top 100 features comparing to randomly select 100 features
log_reg_classify = LogisticRegressionCV(penalty='l2',solver='saga',max_iter=4000,cv=5, random_state=0,multi_class='ovr',Cs=1)
X_evaluate_train = np.log2(np.load('F:/mit-micromaster/machinelearning/data/p2_evaluation_reduced/X_train.npy', mmap_mode='r')+1)
Y_evaluate_train = np.load('F:/mit-micromaster/machinelearning/data/p2_evaluation_reduced/Y_train.npy', mmap_mode='r')
X_evaluate_test = np.log2(np.load('F:/mit-micromaster/machinelearning/data/p2_evaluation_reduced/X_test.npy', mmap_mode='r')+1)
Y_evaluate_test = np.load('F:/mit-micromaster/machinelearning/data/p2_evaluation_reduced/Y_test.npy', mmap_mode='r')
log_reg_classify.fit(X2_train, Y_train)
result_unsup = log_reg_classify.score(X2_test,Y_test)
print("Accuracy: %.2f%%" % (result_unsup * 100.0))

weights = np.sum(np.abs(log_reg_classify.coef_),axis=0)                 #select the top 100 features
#####################first way to get the top 100
top_weights = sorted(range(len(weights)),key=lambda i:weights[i])[-100:]
print(top_weights)
print(weights[top_weights])
##########################second way to get the top 100
top_features = np.argsort(weights,axis=0)[-100:]                     #descend order:-weights
print(top_features)
print(weights[top_features])

random_weights =  random.sample(range(1,len(weights)),100)      #select features randomly
print(random_weights)

X_variations = X_evaluate_train.std(axis=0)
most_variable = np.argsort(X_variations)[-100:]
print(most_variable)
#
X_random = X_variations[random_weights]
X_top100 = X_variations[top_features]
# plt.hist(X_top100,alpha=0.5,bins=15,color='r',label='top100_features')  #seperate show hist
# plt.hist(X_random,alpha=0.5,bins=15,color='b',label='random100_features')
# plt.gca().set(title="Histogram of variance of features",ylabel='Frequency')
plt.hist([X_top100,X_random],bins=15,color=['r','b'],label=['top100_features','random100_features'])
plt.legend()
plt.xlabel("varaince")
plt.ylabel("number  of features")
plt.show()
########################
###train the model for top 100 features data
log_reg_top = LogisticRegression(penalty='l2',solver='saga',max_iter=4000,random_state=0,multi_class='ovr',C=1)          #practice the model by top 100 features
log_reg_top.fit(X_evaluate_train[:,top_features],Y_evaluate_train)
evaluate_test_top = log_reg_top.score(X_evaluate_test[:,top_features],Y_evaluate_test)
print("the score of top 100 features",evaluate_test_top)
print("Accuracy of top 100 features on evaluate test data: %.2f%%" % (evaluate_test_top * 100.0))

log_reg_rand = LogisticRegression(penalty='l2',solver='saga',max_iter=4000,random_state=0,multi_class='ovr',C=1)      #practice the model by random 100 features
log_reg_rand.fit(X_evaluate_train[:,random_weights],Y_evaluate_train)
evaluate_test_random = log_reg_rand.score(X_evaluate_test[:,random_weights],Y_evaluate_test)
print("the score of random selection:", evaluate_test_random)
print("Accuracy of random 100 features on evaluate test data: %.2f%%" % (evaluate_test_random * 100.0))
#################################
#result:
# the score of top 100 features 0.9043321299638989
# Accuracy of top 100 features on evaluate test data: 90.43%
# the score of random selection: 0.5370036101083032
# Accuracy of random 100 features on evaluate test data: 53.70%
###################################################
