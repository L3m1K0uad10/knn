import numpy as np
import math



class Knn:
    """ 
    data (numpy.ndarray) (n_rows, n_cols) 2 dimension: dataset on which the distance calculation will be done
    initial (numpy.ndarray) (n_cols, ) 1 dimension: observation from which the distance between other observations
             will be calculated
    metric (int): any type 
            0 : euclidean distance
            1 : manhattan distance
            2 : minkowski distance
    k (int): number of selected neighbor
    """
    def __init__(self, data:np.ndarray, labels:np.ndarray, initial:np.ndarray, metric = 0, k = 3):
        self._data = data 
        self._labels = labels
        self._initial = initial
        self._metric = metric 
        self._k = k

        self.__distances = np.zeros((self._data.shape[0], 1))

    def __compute_euclidean_distance(self):
        for i, row in enumerate(self._data):
            sum_ = 0
            for j, col in enumerate(row):
                sum_ += math.pow((col - self._initial[j]), 2)
            self.__distances[i] = math.sqrt(sum_)

        return self.__distances

    def __compute_manhattan_distance(self):
        for i, row in enumerate(self._data):
            sum_ = 0
            for j, col in enumerate(row):
                sum_ += abs(col - self._initial[j])
            self.__distances[i] = sum_

        return self.__distances 

    def compute(self):
        """  
        compute distances
        distances (numpy.ndarray) (n_rows, 1)
        """
        if self._metric == 1:
            distances = self.__compute_manhattan_distance()
        else:
            distances = self.__compute_euclidean_distance()
        
        return distances
    
    def classify(self):
        distances = self.compute()
        matches = np.where((self._data == self._initial).all(axis=1))[0]
        
        nearest_neighbors = {}

        if matches.size > 0: 
            idx = matches[0] 
            new_distances = np.delete(distances, idx, axis = 0)    

            for i in range(self._k):
                min_ = np.min(new_distances)
                matches = np.where((new_distances == min_).all(axis = 1))[0]
                idx = matches[0]
                nearest_neighbors[idx] = [min_, str(self._labels[idx][0])]
                new_distances = np.delete(new_distances, idx, axis = 0)
        else:
            for i in range(self._k):
                min_ = np.min(distances)
                matches = np.where((distances == min_).all(axis = 1))[0]
                idx = matches[0]
                nearest_neighbors[idx] = [min_, str(self._labels[idx][0])]
                distances = np.delete(distances, idx, axis = 0)

        nearest_neighbors_dict = nearest_neighbors
        return nearest_neighbors_dict

    
