# K-means

### Intended for:

* all variables are of quantitative type
* Squared Euclidean distance as the dissimilarity measure

### Algorithm

* Init centroids
* iter until convergence:
  * update each data point's label
  * update centroids
  * check convergence

### Implementation

```python
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target

def randomCent(data, k):
    return data[np.random.choice(data.shape[0],k,replace=False)]

def squareEucliDist(A,B):
    return np.sum(np.power(A-B, 2))

def kMeans(data, k, dist=squareEucliDist, cent=randomCent, ITER = 100, show=True):
    n = data.shape[0]
    labels = np.zeros(data.shape[0], np.int32)
    centroids = cent(data, k)
    converged = False
    iter = 0
    if show:
        plt.scatter(data[:,0],data[:,1])
        plt.ion()
    while not converged:
        iter += 1
        old_centroids = centroids.copy()
        # update labels
        for i in range(n):
            d = np.array([dist(data[i],centroids[j]) for j in range(k)])
            labels[i] = np.argsort(d)[0]
        # update centroids
        for m in range(k):
            centroids[m] = np.mean(data[labels==m],axis=0)
        if show:
            plt.cla()
            for m in range(k):
                tmp = data[labels==m]
                plt.scatter(tmp[:,0], tmp[:,1], marker='x')
            plt.scatter(centroids[:,0],centroids[:,1], marker='o')
            plt.pause(0.2)
        # check converge
        if np.mean(np.power(old_centroids-centroids, 2))==0 or iter==ITER:
            converged = True
    if show:
        plt.ioff()
        plt.show()
    return labels, centroids

data = X[:,[1,3]]
labels, cent = kMeans(data, 3, show=True)

data = X[:,:]
labels, cent = kMeans(data, 3, show=False)
print('Accuracy:',max(np.sum(labels==y), np.sum(labels==(y+3)%3), np.sum(labels==(y+2)%3))/y.shape[0])

```

### sklean

```python
from sklearn.cluster import KMeans

km = KMeans(3).fit(data)
labels, cent = km.labels_, km.cluster_centers_

'''
KMeans()
	n_clusters = 8
	init = 'k-means++'
	n_init = 10 # run idependently for 10 times and select the best
	max_iter = 300
	random_state = None # random seed. eg.set to 0
	algorithm = 'auto'

other uses:
	labels = KMeans(3).fit_predict(data)
	test = km.predict([1,2])
	...
'''
```



### k-means++

select the initial centroids as far as possible:

```python
def ppCent(data, k, dist=squareEucliDist):
    centroids = list(data[np.random.choice(data.shape[0],1)])
    for i in range(1, k):
        D = []
        for j in range(data.shape[0]):
            D.append([dist(data[j],centroids[k]) for k in range(len(centroids))])
        D = np.min(np.array(D),axis=1)
        centroids.append(data[np.argsort(D)[-1]])
    return np.array(centroids)
    
```



#### (Weighted) Kernel k-means

One limit for k-means is that it always cluster by a hyperplane.

So we can use a transfer function $\phi()$ to transfer each data vector into a higher dimension, then apply traditional k-means.

Besides, we can also add weight for each data vector.
$$
\sum_{c=1}^{k}\sum_{a_i \in \pi_c} w_i||\phi(a_i) - m_c||^2
$$
Further more, if we simply expand the $l2$ norm, we notice that we even needn't knowing the explicit form of $\phi()$ to compute this norm.

All we need is a Kernel Matrix $K$, where $K_{ij} = \phi(a_i)\phi(a_j)$

for example, a polynomial $K$ can be defined as:
$$
K_{ij} = (a_i \cdot a_j + c)^d
$$





#### Spherical k-means

Used in document clustering, replace Euclidean Distance with Cosine Distance.

