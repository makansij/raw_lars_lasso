import numpy as np

def gradient(x):
    # gradient/derivative of the squared function is 2 times each variable
    # it's like taking a partial derivative wrt each of the variables
    return np.array(2*x)

x = np.array([1.,0.,-3.])
for _ in range(20):
    eps = np.zeros_like(x)         # create empty array [0, 0, 0]
    index = np.argmin(gradient(x))        # get position of the lowest value in the array 
    eps[index] = 0.1               # go to that position in the empty array above, and replace it with epsilon   
    x += eps                       # add the array with epsilon but otherwise empty, to the original array, x
    print(x)


