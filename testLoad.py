from numpy import load
with load('test.npy') as data:
    a = data

print(a[0])