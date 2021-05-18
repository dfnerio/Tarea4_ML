
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class k_means:
    def __init__(self):
        super().__init__()

    def fit(self, data, k, max_iter):  # returns coordinates and a list of indexes
        centroids = {}

        for i in range(k):
            # initializes with random centroids
            centroids[i] = data[np.random.randint(len(data))]

        classifications = {}

        optimized = False

        for iteration in range(max_iter):

            for i in range(k):
                classifications[i] = []

            for i in range(len(data)):
                distances = [pow(np.linalg.norm(
                    data[i] - centroids[c]), 2) for c in centroids]
                classifications[i] = distances.index(min(distances))

            current_centroids = centroids

            for centroid_i in range(len(centroids)):

                current = []

                for classification_i in range(len(classifications)):
                    if classifications[classification_i] == centroid_i:
                        current.append(data[classification_i])

                current_centroid = current_centroids[centroid_i]
                new_centroid = np.average(current, axis=0)

                if (set(current_centroid) == set(new_centroid)):
                    optimized = True
                    print('OPTIMIZED AT ITERATION: %s' % str(iteration + 1))
                    break

                centroids[centroid_i] = new_centroid

            if optimized:
                break

        classifications = list(classifications.values())
        return centroids, classifications

    def predict(self, data, cent, classifs):
        classifications = {}
        for i in range(len(data)):
            classifications[i] = []

        for i in range(len(data)):
            distances = [pow(np.linalg.norm(
                data[i] - centroids[c]), 2) for c in centroids]
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
prediction = alg.predict(new_points, centroids, classifications)
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
