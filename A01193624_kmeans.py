
# A01193624 Diego Frías Nerio
# A01197164 Javier Alejandro Domene Reyes
# Tarea 4 - KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class k_means:
    classifications = {}
    centroids = {}

    def __init__(self):
        super().__init__()

    def fit(self, data, k, max_iter):  # returns coordinates and a list of indexes

        for i in range(k):
            # initialize with random centroids
            self.centroids[i] = data[np.random.randint(len(data))]

        optimized = False

        for iteration in range(max_iter):

            for i in range(k):
                self.classifications[i] = [] # initialize each classification index with an empty array

            for i in range(len(data)):
                distances = [pow(np.linalg.norm(
                    data[i] - self.centroids[c]), 2) for c in self.centroids] # calculate distance for each example set and centroid
                self.classifications[i] = distances.index(min(distances)) # add the cluster index of the minimum distance to the classifications object

            current_centroids = self.centroids # create a copy of the current centroid for further checking

            for centroid_i in range(len(self.centroids)):

                current = [] # array to keep track of sets in current cluster

                for classification_i in range(len(self.classifications)):
                    if self.classifications[classification_i] == centroid_i:
                        current.append(data[classification_i]) # append corresponding set to current cluster array

                current_centroid = current_centroids[centroid_i]
                new_centroid = np.average(current, axis=0) # get new centroid by averaging the current array

                if (set(current_centroid) == set(new_centroid)): # if new centroid is equal to current centroid (1/3)
                    optimized = True
                    print('OPTIMIZED AT ITERATION: %s' % str(iteration + 1))
                    break # then the model is optimized and can stop (2/3)

                self.centroids[centroid_i] = new_centroid # if not, assign the new centroid and continue (3/3)

            if optimized:
                break

        self.classifications = list(self.classifications.values()) # transform the classifications dict into a list of its values
        return self.centroids, self.classifications

    def predict(self, data):
        classifications = {}
        for i in range(len(data)):
            classifications[i] = []

        for i in range(len(data)):
            distances = [pow(np.linalg.norm(
                data[i] - self.centroids[c]), 2) for c in self.centroids]
            classifications[i] = distances.index(min(distances))

        return list(classifications.values())


# main

# blobs

data = pd.read_csv('/Users/diego/Desktop/Tarea4/datasets/blobs.csv')
X = data.to_numpy()

k = 3
alg = k_means()
centroids, classifications = alg.fit(X, k, 150)

colors = ['r', 'b', 'g']

for cluster in range(k):
    for cl_i in range(len(classifications)):
        if classifications[cl_i] == cluster:
            plt.scatter(X[cl_i][0], X[cl_i][1], color=colors[cluster])

new_points = [[8, -2], [-10, 4]]
prediction = alg.predict(new_points)
print("PREDICTIONS", prediction)
for point_i in range(len(new_points)):
    plt.scatter(new_points[point_i][0], new_points[point_i][1], color=colors[prediction[point_i]])

for i, c in enumerate(centroids):
    plt.scatter(centroids[c][0], centroids[c][1], s=80, color="black")

plt.show()


# moons

data = pd.read_csv('/Users/diego/Desktop/Tarea4/datasets/moons.csv')
X = data.to_numpy()

k = 2
alg = k_means()
centroids, classifications = alg.fit(X, k, 150)

colors = ['r', 'b']

for cluster in range(k):
    for cl_i in range(len(classifications)):
        if classifications[cl_i] == cluster:
            plt.scatter(X[cl_i][0], X[cl_i][1], color=colors[cluster])

for i, c in enumerate(centroids):
    plt.scatter(centroids[c][0], centroids[c][1], s=80, color="black")

plt.show()


# circles

data = pd.read_csv('/Users/diego/Desktop/Tarea4/datasets/circles.csv')
X = data.to_numpy()

k = 2
alg = k_means()
centroids, classifications = alg.fit(X, k, 150)

colors = ['r', 'b']

for cluster in range(k):
    for cl_i in range(len(classifications)):
        if classifications[cl_i] == cluster:
            plt.scatter(X[cl_i][0], X[cl_i][1], color=colors[cluster])

for i, c in enumerate(centroids):
    plt.scatter(centroids[c][0], centroids[c][1], s=80, color="black")

plt.show()
