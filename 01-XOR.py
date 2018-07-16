
import numpy as np, random

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = np.array([[0], [1], [1], [0]])

def sigmoid(x, derive = False):
    if derive == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

class xor:
    def __init__(self, input):
        
        self.inputs = input
        self.length = len(self.inputs)
        self.Li = len(self.inputs[0])
        
        self.Wi = np.random.random((self.Li, self.length))
        self.Wh = np.random.random((self.length, 1))
    
    def wSum(self, input):
        output_1 = sigmoid(np.dot(input, self.Wi))
        output_2 = sigmoid(np.dot(output_1, self.Wh))
        return output_2
    
    def train(self, input, output):
        for x in range(1):
            
            inputs = input
            layer_1 = sigmoid(np.dot(input, self.Wi))
            layer_2 = sigmoid(np.dot(layer_1, self.Wh))
            
            
            l2_error = output - layer_2
            l2_delta = np.multiply(l2_error, sigmoid(layer_2, derive = True))
            
            l1_error = np.dot(l2_delta, self.Wh.T)
            l1_delta = np.multiply(l1_error, sigmoid(layer_1, derive = True))
            
            
            self.Wh += np.dot(layer_1, l2_delta)
            self.Wi += np.dot(inputs.T, l1_delta)

mlp = xor(inputs)

mlp.train(inputs, output)
print("Trained output XOR: \n")
print(mlp.wSum(inputs))
            
            
        
        
