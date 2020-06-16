#### cluster

* classes
* functions

#### datasets

* loaders

  * load_*

  ```python
  load_digits(n_classes=10, return_X_y=False)
  digits = load_digits()
  '''
  digits.data : [N, 64]
  digits.images : [N, 8, 8]
  digits.target : [N, 1]
  '''
  data, target = load_digits(return_X_y=True)
  
  load_iris(return_X_y=False)
  ```

  * fetch_*

  ```python
  # fetch_lfw_people(data_home=None, resize=0.5, min_faces_per_persion=0, color=False, ...)
  lfw = fetch_lfw_people()
  '''
  '~/scikit_learn_data' is the data_home by default.
  '''
  ```

* generators

#### decomposition

#### ensemble

```python
RandomForestRegressor()
RandomForestClassifier()
```



#### feature_extraction

####  linear_model

#### manifold

```python


```



#### metrics

* classification metrics

  ```python
  classification_report(y_true, y_pred)
  '''
  @ labels=None : use in report
  @ digits=2 : precision of numbers
  # string of report
  '''
  confusion_matrix(yt, yp)
  '''
  # confusionMat : [N_classes,N_classes]
  	this mat count the result of all classification.
  	>>> y_true = [2, 0, 2, 2, 0, 1]
  	>>> y_pred = [0, 0, 2, 2, 0, 2]
  	>>> confusion_matrix(y_true, y_pred)
  	array([[2, 0, 0],
        	   [0, 0, 1],
         	   [1, 0, 2]])
  '''
  ```


#### model_selection

* Splitter classes

* Splitter functions

  ```python
  train_test_split(*arr, ...)
  '''
  $ *arrays: the whole dataset. (X) or (X,y)
  $ test_size: float for percentage, int for number. default is 0.25
  $ train_size: 1 - test_size
  $ shuffle=True
  # splitting: (X_train, X_test[, y_train, y_test])
  '''
  cross_val_score(clf, X, y, cv=3)
  '''
  simply print this to get cv.
  '''
  kf = KFold(n_splits=3, shuffle=False)
  '''
  for train_idx, test_idx in kf.split(X):
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = ...
  '''
  
  ```

* Hyper parameter optimizer

  ```python
  GridSearchCV(estimator, ...)
  '''
  $ estimator: interface, have a score funtion.
  $ param_grid: dict or list
  @ cv_results_: dict of ndarray, or dataframe
  @ best_estimator_: 
  @ best_score_:
  @ best_params_:
  M fit(X[, y])
  M predict(X): use the best.
  
  svc = svm.SVC()
  parameters = {'kernel':['linear', 'rbf'], 'C':[1, 10]}
  clf = GridSearchCV(svc, parameters)
  '''
  ```

* Model validation

  ```python
  cross_validate()
  cross_val_score(estimator, X, y=None)
  '''
  $ cv=None: int of KFold. default is 3.
  # scores
  '''
  cross_val_predict()
  ```


#### naive_bayes

#### pipeline

#### preprocessing

```python
scaler = StandardScaler()
df[col+"_scaled"] = scaler.fit_transform(df[col].values.reshape(-1,1))

```



#### svm

#### tree

#### utils

