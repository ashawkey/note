#### Introduction

1. Machine Learning

* Supervised learning
  * classification
  * regression
* Unsupervised learning
  * clustering
  * density estimation

2. Loading example dataset

   ```python
   from sklearn import datasets
   iris = datasets.load_iris()
   digits = datasets.load_digits()
   
   print(digits.data) # [S, N*M]
   print(digits.data.shape) # [N, M]
   print(digits.target) # [S, 1]
   print(digits.images) # [S, N, M]
   
   ```

3. Learning and predicting

   ```python
   from sklearn import svm
   # build a classifier(estimator)
   clf = svm.SVC(gamma=0.001, C=100.)
   # fit in training data
   clf.fit(digits.data[:-1], digits.target[:-1])
   # test with the last data
   clf.predict(digits.data[-1])
   ```


4. Model persistence

   ```python
   import pickle
   s = pickle.dumps(clf)
   clf2 = pickle.loads(s)
   # or
   from sklearn.externals import joblib
   joblib.dump(clf, "file.pkl")
   clf = joblib.load("file.pkl")
   ```

#### Supervised Learning

* k-NN algorithm for classification

  one of the simplest algorithm for ML. 

  **Prediction is just the vote of its k-NN.**

  ```python
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier()
  knn.fit(iris_X_train, iris_y_train)
  knn.predict(iris_X_test)
  ```

  > the curse of dimension:
  >
  > The common theme of these problems is that when the dimensionality increases, the [volume](https://en.wikipedia.org/wiki/Volume) of the space increases so fast that the available data become sparse. This sparsity is problematic for any method that requires statistical significance. In order to obtain a statistically sound and reliable result, the amount of data needed to support the result often grows exponentially with the dimensionality

* Linear model

  * Linear regression model:

  $$
  y = Xw + \epsilon \\
  J(w) = min_w||Xw-y||^2 \\
  \nabla J(w) = 2X'(Xw-y) \\
  w_k = w_{k-1} - \alpha\nabla J(w) \\
  in \ fact \ we \ can \ solve \ it: \\
  w = (X'X)^{-1}X'y
  $$

  ```python
  from sklearn import linear_model
  regr = linear_model.LinearRegression()
  regr.fit(X_train, y_train)
  # learned \beta values
  print(regr.coef_)
  # variance score. 1 is perfect.
  regr.score(X_test, y_test)
  ```

  * Ridge regression model

    LR is very **sensitive to noise**, so RR is designed to improve this.
    $$
    J(W) = min_w\{||Xw-y||^2 + \alpha||w||^2\}
    $$
    so we can set an $\alpha$ to avoid $w$ being so big and thus too sensitive.

    $\alpha||w||^2$ is called the regularization.

    ```python
    regr = linear_model.Ridge(alpha=0.1)
    # also, if we use L1 norm:
    regr = linear_model.Lasso(alpha=0.1)
    ```

  * Logistic regression model

    linear model proposed for Classification.

    C is for regularization.

    ```python
    logistic = linear_model.LogisticRegression(C=1e5)
    ```

* SVM 

  * Linear SVM

    SVM can be used in SVR(regression) or SVC(classification) form.

    ```python
    from sklearn import svm
    svc = svm.SVC(kernel='linear')
    ```

  * Kernels

    ```python
    svc = svm.SVC(kernel='rbf')
    '''
    rbf: exp(-gamma||x-x'||^2)
    poly: (<x,x'> + r)^d
    sigmoid:
    '''
    ```


#### Model Selection

* Score

  **Bigger is Better.**

  * K Fold Cross validation

    ```python
    from sklearn.model_selection import KFold, cross_val_score
    # split to 3 part
    k_fold = KFold(n_splits=3)
    train_indices, test_indices = k_fold.split(X)
    # KFold(n_splits, shuffle=False)
    ```

  * cross_val_score

    ```python
    cross_val_score(estimator, X, y)
    # by default using 3-fold, and output three scores.
    ```

  * CV estimators

    some shortcut for calling CV automatically.

    ```python
    lasso = linear_model.lassoCV()
    lasso.fit(...)
    print(lasso.alpha_)
    ```

* Grid search

  to search through the parameter space and find the optimal.

  ```python
  from sklearn_model_selection import GridSearchCV
  Cs = np.linspace(-6, -1, 10)
  clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1) 
  # n_jobs=-1 : use all cpus
  # by default, it use 3-fold Cross Validation.
  clf.fit(train_X,train_y)
  print(clf.best_score_)
  print(clf.best_estimator_.C)
  clf.score(test_X, test_y)
  ```


#### Unsupervised learning

* Clustering

  * K-Means

  ```python
  from sklearn import cluster
  km = cluster.KMeans(n_clusters=3)
  km.fit(X)
  print(km.labels_)
  ```

* Decomposition

  * PCA (Principle)

  ```python
  from sklearn import decomposition
  pca = decomposition.PCA()
  pca.fit(X)
  # larger score means more useful
  print(pca.explained_variance_)
  # choose targeted dims, here 2
  pca.n_components = 2
  X_reduced = pca.fit_transform(X)
  
  
  ```

  * ICA (Independent)

    separate a multivariate signal into additive subcomponents that are maximally independent.



