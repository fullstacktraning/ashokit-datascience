import numpy as np

inputs = np.array([1.0,2.0,3.0])

weights = np.array([[0.1,0.2,-0.5],
                    [0.5,-0.91,0.26],
                    [-0.26,-0.27,0.17]])

biases = np.array([1.0,2.0,3.0])

output = np.dot(inputs,weights) + biases

def relu(output):
    return np.maximum(0,output)

res = relu(output)
print(res)