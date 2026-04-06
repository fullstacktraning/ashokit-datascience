import numpy as np

inputs = np.array([1.0,2.0,3.0])

weights = np.array([0.2,0.8,-0.5]) # back propagation / decent gradient

bias = 2.0

output = np.dot(inputs,weights) + bias

def relu(output):
    return np.maximum(0,output)

res = relu(output)
print(res)

# output < 0 --->. 0
# output > 0 ---> output