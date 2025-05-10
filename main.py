import numpy as np
import pandas as pd

from knn import Knn



df = pd.read_csv("iris/iris.csv")
data = df.to_numpy()

rows, cols = data.shape

# splitting label from futures
labels = data[0:rows + 1, cols - 1:]
data = data[0:rows + 1, 0:cols - 1]

# training model data and testing model data
# let's reserve 2 observations only for testing
test_labels = np.array([labels[0], labels[rows - 1]])
test_data = np.array([data[0], data[rows - 1]])

labels = np.delete(labels, 0, axis = 0)
labels = np.delete(labels, rows - 1 - 1, axis = 0) # retrieve one more because of row at 0 removed
data = np.delete(data, 0, axis = 0)
data = np.delete(data, rows - 1 - 1, axis = 0) # retrieve one more because of row at 0 removed


""" select the value of k such that k > no_classes """
model = Knn(data, labels, test_data[0], 1)
distances = model.compute()
#print(distances)
nn = model.classify()
print("\n", nn)
