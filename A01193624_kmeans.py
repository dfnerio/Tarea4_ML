
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class k_means:
    def __init__(self):
        super().__init__()

    def fit(self, data, k, max_iter):  # returns coordinates and a list of indexes
        centroids = {}

        for i in range(k):
            centroids[i] = data[i]

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
        dists = [np.linalg.norm(data - cent[c]) for c in cent]
        classif = dists.index(min(dists))
        return classif


# main

data = pd.read_csv('datasets/blobs.csv')
X = data.to_numpy()

k = 4
alg = k_means()
centroids, classifications = alg.fit(X, k, 150)

print("CLASSIFICATIONS", classifications)
print("CENTROIDS", centroids)

colors = ['k', 'b', 'g', 'r']

# plt.scatter(centroids, y=classifications)
# plt.scatter()
# plt.ylabel('some numbers')
# plt.show()

new_point = [-8, 2]
cluster = alg.predict(new_point, centroids, classifications)
plt.scatter(new_point[0], new_point[1], color = colors[cluster])

for i, c in enumerate(centroids):
    plt.scatter(centroids[c][0] , centroids[c][1] , s = 80, color = colors[i])

plt.show()
