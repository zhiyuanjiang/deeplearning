import numpy as np

# np.clip(a, min_value, max_value), 将元素值都控制在min_value~max_value
a = np.array([[1,2,3],[4,5,6]])
print(a)
b = np.clip(a, 2, 4)
print(b)

# np.random.choice()
np.random.seed(0)
p = np.array([0.1, 0., 0.7, 0.2])
a = np.random.choice([1, 2, 3, 4], p=p.ravel())
print(a)

# np.ravel()
a = np.array([1,2,3])
print(a.ravel())

######
a = [None]+[x for x in range(4)]
print(a)

#####
for x in reversed(range(10)):
    print(x)

#####
with open('F:\\deeplearning-data\\word2vec-data\\glove.txt') as file:
    a = file.readline()

a = np.array([1, 2, 4, 1])
print(np.argmax(a))