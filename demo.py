from model import NeuralNetwork
import numpy as np
import matplotlib as plt

# random vars for x and y
num_samples = 100 #'rows' of the dataset
num_features = 2 # 'cols' of the dataset
X = np.random.randn(num_samples, num_features)
Y = np.random.randn(num_samples)
print(X)
print (X.shape)
print(Y)
print (Y.shape)

# testing NN on data generated above
model = NeuralNetwork(X = X, y = Y, num_epochs = 100)
model_res = model.train()
