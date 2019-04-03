import numpy as np
array_int = np.array([1,2,3])
array_float = array_int.astype("float64")
array_float

np.arange(10).shape

np.ones([10,2])
np.zeros([10,3])
np.ndarray(10).reshape(2,-1)

array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
array5 = array3d.reshape(-1,1)
print('array4:\n', array5.tolist())

array1 = np.arange(start=1, stop=10)
value = array1[2]
value

org_array = np.array([3,1,9,5])
np.sort(org_array)
org_array.sort()

sort_array1_desc = np.sort(org_array)[::-1]
sort_array1_desc

array2d = np.array([[8,12],[7,1]])
sort_array2d_axis0 = np.sort(array2d, axis=0)
sort_array2d_axis0

sort_array2d_axis1 = np.sort(array2d, axis=1)
sort_array2d_axis1

org_array = np.array([3,1,9,5])
sort_indices = np.argsort(org_array)[::-1]
type(sort_indices)
org_array[sort_indices]

A = np.array([[1,2,3],
             [4,5,6]])
B = np.array([[ 7, 8],
             [ 9,10],
             [11,12]])

np.dot(A, B)

A = np.array([[1,2],
              [3,4]])
np.transpose(A)

np.transpose(np.arange(10))