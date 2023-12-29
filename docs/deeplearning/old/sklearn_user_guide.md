### Supervised learning

* Generalized Linear Models

  * Ordinary Least Squares
    

$$
\displaylines{

    argmin_w\{||Xw+b-y||_2^2\}
    
}
$$


    ```python
    clf = LinearRegression()
    clf.fit(X, y) # [N, M]
    clf.coef_ # w, [M, 1]
    clf.intercept_ # b
    ```

  * Ridge Regression
    

$$
\displaylines{

    argmin_w\{||Xw+b-y||_2^2+\alpha||w||_2^2\}
    
}
$$


    ```python
    clf = Ridge(alpha=0.5)
    '''
    $ alpha: regularization, positive or 0. 
    '''
    ```

  * Lasso Regression
    

$$
\displaylines{

    argmin_w\{\frac{1}{2N}||Xw+b-y||_2^2 + \alpha||w||_1\}
    
}
$$


  * Least Angle Regression

    ```python
    clf = LassoLars(alpha=.1)
    ```

  * Bayesian Regression

  * Logistic Regression

    for classification instead of regression.

    we can use L1 or L2 norm for regularization.
    

$$
\displaylines{

    argmin_{w,b}\{\frac 1 2 ||w||_2 + C\sum_{i=1}^nlog(exp(-y_i(X_i^Tw+b))+1)\}
    
}
$$


    ```python
    ...
    ```

  * SGD

    for large dataset

  * Perceptron

    for large dataset

  * Robustness regression

    detect outliers

  * Polynomial Regression

    by adding dimensions, we turn a polynomial model into linear.

    ```python
    from sklearn.preprocessing import PolynomialFeatures
    X = np.arange(6).reshape(3, 2)
    # [x1, x2]
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X)
    # [1, x1, x2, x1^2, x1x2, x2^2]
    ```

* Linear and Quadratic Discriminant Analysis

* Kernel ridge regression

  faster than SVM in large dataset with near performance.

* Support Vector Machines

  * 

### Unsupervised learning

### Model selection and evaluation

### Dataset transformations

### Dataset loading utilities

### Scaling & Performance

