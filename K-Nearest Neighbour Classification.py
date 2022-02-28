#!/usr/bin/env python
# coding: utf-8

# # ML ASSIGNMENT 

# IMPORTING ALL THE NECESSAY LIBRARIES

# In[26]:


import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from collections import Counter


# # IMPORTING DATASET (CIFAR-10) 

# In[27]:


from keras.datasets import cifar10
(x_train_copy, y_train_copy), (x_test_copy, y_test_copy) = cifar10.load_data()
print('x_train shape:', x_train_copy.shape)
print('y_train shape:', y_train_copy.shape)
print(x_train_copy.shape[0], 'train samples')
print(x_test_copy.shape[0], 'test samples')


# # ADJUSTING NUMBER OF INPUT/OUTPUT TO USE

# In[28]:


train_ex=15000
test_ex=2000


# In[29]:


x_train=x_train_copy[:train_ex]
x_test=x_test_copy[:test_ex]


# In[30]:


y_train=y_train_copy[:train_ex]
y_test=y_test_copy[:test_ex]


# # TRAIN DATA

# In[31]:


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# # TEST DATA

# In[32]:


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 5

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_test == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# # DATA PREPROCESSING
# 
# 1. Reshaping the dataset from 3-D matrix to 1-D vector.
# 2. Standardizing the dataset

# In[33]:


x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape, x_test.shape)


# In[34]:


x_train=x_train/255.0
x_test=x_test/255.0

x_train


# # KNN FROM SCRATCH (ALGORITHM)

# EUCLIDEAN DISTANCE OF A TEST POINT FROM ALL TRAIN POINT

# In[35]:


def euclidean_distance(train,test):

    diff=test-train
    dist=np.sqrt(np.sum(diff**2))    
    return dist


# In[36]:


def knn_from_scratch(x_test, x_train, y_train,k):

    y_hat=[]
    for test in x_test:
        distances=[]
        for train in x_train:

            dist=euclidean_distance(train,test)
            distances.append(dist)


        y_train=y_train.reshape(y_train.shape[0],)
        data={
            'Distances':distances,
            'class':y_train
        }

        df_distance=pd.DataFrame(data)
        df_k_distance=df_distance.sort_values(by=['Distances'],axis=0)[:k]
        counter = Counter(y_train[df_k_distance.index])
        prediction=counter.most_common()[0][0]

        y_hat.append(prediction)
        
    return y_hat


# # IMPLEMENTATION PART 
# **USING K-FOLD CROSS VALIDATION**

# In[ ]:


from sklearn.metrics import accuracy_score

possible_k=[1,3,5,7,9,11,13,15,17,19,21]
y_hat=[]
accuracies=[]

#for k-fold cross-validation
#----------------------------------#
num_folds=5

x_train_folds=np.array_split(x_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)

#----------------------------------#


for k in possible_k:
    avg_accuracy=0
    for i in range(num_folds):
        
        x_val=x_train_folds[i]
        y_val=y_train_folds[i]
        x_train=x_train_folds
        y_train=y_train_folds
        
        temp = np.delete(x_train,i,0)
        x_train = np.concatenate((temp),axis = 0)
        y_train = np.delete(y_train,i,0)
        y_train = np.concatenate((y_train),axis = 0)
        
        y_hat= knn_from_scratch(x_val,x_train,y_train,k)
        
        avg_accuracy+=accuracy_score(y_val, y_hat)/num_folds
        
    accuracies.append(avg_accuracy)


# # VAL ACCURACY v/s K (GRAPH)

# In[ ]:


plt.plot(possible_k, accuracies,color='black', linestyle='dashed', linewidth = 2, 
         marker='o', markerfacecolor='blue', markersize=9) 

plt.xlabel('K values',labelpad=15)
plt.ylabel('Validation Accuracy',labelpad=20)
plt.title('Validation Accuracy for different Values of K')
plt.show()


# # BEST K AND TEST ACCURACY

# In[20]:


best_k=15
y_hat_test=knn_from_scratch(x_test,x_train,y_train,best_k)
print(accuracy_score(y_test, y_hat_test))


# # TEST ERROR

# In[21]:


from sklearn.metrics import  mean_absolute_error,mean_squared_error

print(" TEST Mean Absolute Error= ", mean_absolute_error(y_test, y_hat_test)) 
print("\n TEST Mean Squared Error= ", mean_squared_error(y_test, y_hat_test))


# # CONFUSION MATRIX

# In[22]:


from sklearn.metrics import confusion_matrix
print("\n",confusion_matrix(y_test,y_hat_test))


# # **PRECISION v/s RECALL**

# CLASSIFICATION REPORT

# In[23]:


from sklearn.metrics import classification_report

print("\n",classification_report(y_test, y_hat_test))


# **PRECISION AND RECALL CURVE FOR EACH CLASS**

# In[ ]:


from sklearn.preprocessing import label_binarize

y_test= label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
y_hat_test= label_binarize(y_hat_test, classes=[0,1,2,3,4,5,6,7,8,9])


# In[ ]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precision = dict()
recall = dict()
for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_hat_test[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()


# In[ ]:




