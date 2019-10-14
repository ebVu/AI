import pandas as pd 

#1 argsort() method in pandas
"""
df = pd.Index([17, 69, 33, 5, 11, 74, 10, 5])
print(df)
print(df.argsort())
print(df[df.argsort()])
"""
#2 notation about array
"""
a[start:stop]  # items start through stop-1
a[start:]      # items start through the rest of the array
a[:stop]       # items from the beginning through stop-1
a[:]           # a copy of the whole array
a[start:stop:step] # start through not past stop, by step
a[-1]    # last item in the array
a[-2:]   # last two items in the array
a[:-2]   # everything except the last two items


It's pretty simple really:

a[start:stop]  # items start through stop-1
a[start:]      # items start through the rest of the array
a[:stop]       # items from the beginning through stop-1
a[:]           # a copy of the whole array

There is also the step value, which can be used with any of the above:

a[start:stop:step] # start through not past stop, by step

The key point to remember is that the :stop value represents the first value that is not in the selected slice. So, the difference between stop and start is the number of elements selected (if step is 1, the default).

The other feature is that start or stop may be a negative number, which means it counts from the end of the array instead of the beginning. So:

a[-1]    # last item in the array
a[-2:]   # last two items in the array
a[:-2]   # everything except the last two items

Similarly, step may be a negative number:

a[::-1]    # all items in the array, reversed
a[1::-1]   # the first two items, reversed
a[:-3:-1]  # the last two items, reversed
a[-3::-1]  # everything except the last two items, reversed

#Example:
a = [1, 2, 3, 4, 5]
#index of a [0, 1, 2, 3, 4]
print(a[1::-1]) # index 1 of a array is start, step = -1 => index: 1 -> 0
print(a[:1:-1]) # index 1 of a array is stop, step = -1 => index: 4 -> 3 -> 2
"""
#3 as_matrix method in Pandas
"""
#Example:
  
# Creating the Series 
sr = pd.Series(['New York', 'Chicago', 'Toronto', 'Lisbon', 'Rio']) 
  
# Create the Index 
index_ = ['City 1', 'City 2', 'City 3', 'City 4', 'City 5']  
  
# set the index 
sr.index = index_ 
  
# Print the series 
print(sr) 


# return numpy array representation 
result = sr.as_matrix() 
  
# Print the result 
print(result) 
"""
#4 reshape() method in Numpy
#Example
"""
import numpy as geek 
  
array = geek.arange(8) 
print("Original array : \n", array) 
  
# shape array with 2 rows and 4 columns 
array = geek.arange(8).reshape(2, 4) 
print("\narray reshaped with 2 rows and 4 columns : \n", array) 
  
# shape array with 2 rows and 4 columns 
array = geek.arange(8).reshape(4 ,2) 
print("\narray reshaped with 2 rows and 4 columns : \n", array) 
  
# Constructs 3D array 
array = geek.arange(8).reshape(2, 2, 2) 
print("\nOriginal array reshaped to 3D : \n", array) 
"""

# Python program explaining  
# where() function  

import numpy as np 
  
# a is an array of integers. 
a = np.array([[1, 2, 3], [4, 5, 6]]) 
  
print(a) 
  
print ('Indices of elements <4') 
  
b = np.where(a<4) 
print(b) 
  
print("Elements which are <4") 
print(a[b]) 

2
	
# Create a numpy array from list
arr = np.array([11, 12, 13, 14, 15, 16, 17, 15, 11, 12, 14, 15, 16, 17])
# pass condition expression only
result = np.where((arr > 12) & (arr < 16))
print('hello')
print(arr[result])

"""
# Python | Extracting rows using Pandas .iloc[]
path = 'D:\\AI\\Python\\nba.csv'


# importing pandas package 
import pandas as pd 
  
# making data frame from csv file  
data = pd.read_csv(path) 
  
# retrieving rows by loc method  
row1 = data.iloc[[4, 5, 6, 7]] 
  
# retrieving rows by loc method  
row2 = data.iloc[4:8] 
  
# comparing values 
row1 == row2 
"""

#5 unique method in numpy.unique
#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.unique.html


#6 Tensorflow: placeholder
#https://databricks.com/tensorflow/placeholders

#6 Tensorflow: bias
#https://www.quora.com/What-is-bias-in-artificial-neural-network

# tride in CNN
#https://medium.com/machine-learning-algorithms/what-is-stride-in-convolutional-neural-network-e3b4ae9baedb

#kernels in ML
#https://towardsdatascience.com/kernel-function-6f1d2be6091

#CNN for computer vision, image
#https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d

#axis in tensor
import tensorflow as tf
x = tf.constant([0, 1, 2])

A=tf.constant([2,20,30,3,6]) # Constant 1-D Tensor
tf.math.argmax(A) # output 2 as index 2 (A[2]) is maximum in tensor A
B=tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])

#B have 2 axes, axis 0 is colum, axis is row
tf.math.argmax(B,0) # [2, 2, 0, 2, 2]
tf.math.argmax(B,1) # [2, 2, 1]

print('hello')