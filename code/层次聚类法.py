import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
def getLinkageMat(model):
    children = model.children_
    cs = np.zeros(len(children))
    N = len(model.labels_)
    for i,child in enumerate(children):
        count = 0
        for idx in child:
            count += 1 if idx < N else cs[idx - N]
        cs[i] = count
    return np.column_stack([children, model.distances_, cs])

X, y = make_blobs(n_samples=50)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
mat = getLinkageMat(model)
test = dendrogram(mat, truncate_mode='level', p=3)
# test = dendrogram(mat)
plt.show()
