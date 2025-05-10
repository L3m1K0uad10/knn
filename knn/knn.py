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
        
        if matches.size > 0: 
            idx = matches[0] 
            new_distances = np.delete(distances, idx, axis = 0)
            min_ = np.min(new_distances)

            nearest_neighbors = {}
            count = 0
            for i, row in enumerate(new_distances):
                if row[0] <= min_:
                    nearest_neighbors[i] = [float(row[0]), str(self._labels[i][0])]
                    count += 1
                    new_distances = np.delete(new_distances, i, axis = 0)
                    min_ = np.min(new_distances)
                if count == self._k:
                    break
        else:
            min_ = np.min(distances)

            nearest_neighbors = {}
            count = 0
            for i, row in enumerate(distances):
                print("---", row[0], "---", i)
                if row[0] <= min_:
                    print("+++", row[0])
                    nearest_neighbors[i] = [float(row[0]), str(self._labels[i][0])]
                    count += 1
                    distances = np.delete(distances, i, axis = 0)
                    min_ = np.min(distances)
                    print(min_)
                if count == self._k:
                    break

        nearest_neighbors_dict = nearest_neighbors
        return nearest_neighbors_dict

    
