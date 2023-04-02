import numpy
import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


#                               File System
file1 = open("G:/Hunain.txt","r")
print(file1.read())
file1.close()
print(file1.closed)

with open("G:/Hunain.txt","r") as file1:
    print(file1.read())
print(file1.closed)

with open("G:/Hunain.txt","r") as file1:
    file_stuff = file1.readlines(2)
    print(file_stuff)

with open("G:/Hunain.txt","w") as file1:
    file1.write("Hey Charles How are you man.")

# ================================================================================================

x = {"student":["David","Samuel","Terry","Evan"],
     "Age":[27,24,22,32],
     "Country":["UK","Canada","China","USA"],
     "Course":["Python","Data Structures","Machine Learning","Web Development"],
     "Marks":[85,72,89,76]}
df = pd.DataFrame(x)
print(df)

b = df["Marks"]
print(b)

c = df[["Country","Course"]]
print(c)

# ----------------------------Numpy--------------------------------------------
# It is a library for scientific computations

# Basic Numpy functions
a = np.array([1, 2, 3, 4, 5])
print(type(a))
print(a.dtype)
print(a.size)   # This will represent the size of the numpy array.
print(a.ndim)   # This will represent number of directions
print(a.shape)  # This will represent the size of the array in every direction


# Numpy indexing and Slicing
c = np.array([1, 2, 3, 4, 5])
c[0] = 100
c[4] = 500
c[1:4] = 200,300,400
print(c)
d = c[1:4]
print(d)
print(type(d))

# Numpy Basic Operations

# .......................................
# (Vector Addition)
u = np.array([1,0]) # Here u1 = 1, u2 = 0    (Concept)
v = np.array([0,1]) # Here v1 = 0, v2 = 1    (Concept)
z = u+v             # So u1+v1,u2+v2 = z1,z2 (Concept)
print(z)

# Comparison with normal array addition......
u = [1,0]
v = [0,1]
z = u+v
print(z)

# .......................................

# (Vector Subtraction)
u = np.array([1,0]) # Here u1 = 1, u2 = 0                                 (Concept)
v = np.array([0,1]) # Here v1 = 0, v2 = 1                                 (Concept)
z = u-v             # So (u1-v1 = 1-0),(u2-v2 = 0-1) = (z1 = 1),(z2 = -1) (Concept)
print(z)

# (Numpy Array multiplication with scaler)
Y = np.array([1,2])
z = 2*Y
print(z)

# (Product of two Numpy arrays)
u = np.array([1,2])
v = np.array([2,3])
z = u*v
print(z)

# (Dot Product)
u = np.array([1,2])  # u1 = 1, u2 = 2
v = np.array([3,1])  # v1 = 3, v2 = 1
z = np.dot(u,v)      # z  = (u1v1 + u2v2)
print(z)

# (Adding constant to Numpy array)
u = np.array([1,2,3,-1])
z = u + 1
print(z)

# Universal functions

# Mean : The mean (average) of a data set is found by adding all numbers
# in the data set and then dividing by the number of values in the set.
# (mean value)
a = np.array([1,-1,1,-1])
how_mean = a.mean()
print(how_mean)

# (Maximum value)
a = np.array([1,-2,3,4,5])
how_max = a.max()
print(how_max)

# (Minimum value)
a = np.array([1,-2,3,4,5])
how_min = a.min()
print(how_min)

# (Sin Function)
np.pi            # it will demonstrate the value of pi
x = np.array([0, np.pi/2, np.pi])
y = np.sin(x)
print(y)

# (linspace Function)
a = np.linspace(-2,2,num = 5) # This function shows the intervals between the given value.
print(a)

# Plotting mathamatical functions

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
print(y)
plt.plot(x,y)
plt.show()




# have to understand the whole thing....

def Plotvec1(u, z, v):
    ax = plt.axes()  # to generate the full window axes
    ax.arrow(0, 0, *u, head_width=0.05, color='r',
             head_length=0.1)  # Add an arrow to the  U Axes with arrow head width 0.05, color red and arrow head length 0.1
    plt.text(*(u + 0.1), 'u')  # Adds the text u to the Axes

    ax.arrow(0, 0, *v, head_width=0.05, color='b',
             head_length=0.1)  # Add an arrow to the  v Axes with arrow head width 0.05, color red and arrow head length 0.1
    plt.text(*(v + 0.1), 'v')  # Adds the text v to the Axes

    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')  # Adds the text z to the Axes
    plt.ylim(-2, 2)  # set the ylim to bottom(-2), top(2)
    plt.xlim(-2, 2)  # set the xlim to left(-2), right(2)


# (Numpy multiplication)
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([2, 1, 2, 3, 4, 5])
print(np.multiply(arr1,arr2))



# 2D numpy

A = np.array([[11,12,13],[21,22,23],[31,32,33]])
a = A.ndim
print(a)

A = np.array([[11,12,13],[21,22,23],[31,32,33]])
a = A.shape
print(a)

# Slicing in @d numpy array
A = np.array([[11,12,13],[21,22,23],[31,32,33]])
a = A[0,0:3]
print(a)

# addition will be the same as 2d...corresponding elements will be added to each other like a matrix.
# Multiplication will on;y be done if the rows of 1st will be same as rows of 2nd matrix.

A = np.array([[11,12,13],[21,22,23],[31,32,33]])
a = A[0][0:2]
print(a)

# Practice Questions

X=np.array([[1,0],[0,1]])
Y=np.array([[0,1],[1,0]])
Z=X+Y
print(Z)

# Question 4
# What is the value of  Z after the following code is run?

X=np.array([[1,0],[0,1]])
Y=np.array([[2,2],[2,2]])
Z=np.dot(X,Y)
print(Z)