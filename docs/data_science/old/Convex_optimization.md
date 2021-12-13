# Optimization

### Mathematical optimization

Optimization arises everywhere.

But most of them are intractable.

Exception is the tractable Convex optimization.



### Convex optimization

##### Problem definition (standard form)

$$
minimize \ f_0(x) \\
subject \ to \ f_i(x) \le 0, i=1,...,m \\
Ax=b
$$

- $x \in R^n$
- equality constraints are linear
- $f_0, ..., f_m$ are convex.



### Solvers

##### CVXPY

```python
from cvxpy import *

x = Variable(n)
cost = sum_squares(A*x-b) + gamma*norm(x,1)
prob = Problem(Minimize(cost), [norm(x, "inf")<=1])
opt_val = prob.solve()
solution = x.value
```







# Applications

### Portfolio Optimization



### Regression variation



### Model fitting

##### Regularized loss minimization

* Regression Problem
  $$
  R^n \rightarrow R\cup\{\infty\}
  $$

* Regularized Loss

  m examples, each has n dimensions.
  $$
  (1/m)\sum_i^nL(x_i, y_i, \theta) + r(\theta)
  $$
  ![1548743843595](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1548743843595.png)

  * $\lambda > 0$ scales regularization.
  * all lead to convex fitting problems.



# Constructive Convex Analysis & DCP

### Convex Optimization

##### Conic form

$$
minimize\ c^Tx \\
subject\ to\ Ax=b,\ x\in K
$$

- $x \in  R^n$
- $K$ is convex cone.
- linear objective, equality constraints.

### How to solve a convex optimization problem?

##### Curvature

- convex (凹)
  - definition: $f(\theta x+(1-\theta)y) \le \theta f(x)+(1-\theta)f(y)$
  - $\nabla^2f(x)\ge0$
  - can be constructed using basic functions that are convex or concave, and using transformations that preserve convexity.
- concave (凸)
  - $-f(x)$ is convex
- affine
  - both concave and convex.
  - has the form $f(x) = a^Tx+b$



##### Basic convex functions

- $x^p,\ p\ge1\ or\ p\le 0$
- $e^x$
- $xlogx$
- $a^T+b$, affine
- $x^TPx,\ P\ge0$
- $||x||$
- $max(x_1, x_2, ..)$

Less basic ones:

- $\frac {x^2} {y}, y>0$, jointly convex for x and y.
- $xlog(x/y)$, jointly convex for x and y. 
- $x^TY^{-1}x,\ Y\ge0$
- $log(e^{x_1}+e^{x_2}+...)$
- sum of largest k entries
- $\lambda_{max}(X), X=X^T$



##### Basic concave functions

- $x^p,\ 0 \le p\le 1$
- $logx$
- $x^TPx,\ P\le0$
- $min(x_1, x_2, ...)$

Less basic ones:

- $log\ det\ X$
- $\lambda_{min}(X), X=X^T$



##### Calculus rules that keeps convexity

- nonnegative scaling

- sum

- affine composition

- pointwise maximum (non-differentiability)

- **general composition rule:**

  $h(f_1(x), ..., f_k(x))$ is convex when $h$ is convex and for each $i$:

  - $h$ is increasing in argument $i$, and $f_i$ is convex, or
  - $h$ is decreasing in argument $i$, and $f_i$ is concave, or
  - $f_i$ is affine



eg. show the following function is convex:
$$
f(u, v) = (u+1)log(\frac {u+1}{min(u,v)})
$$

##### Constructive Convexity verification

view the function as an expression tree, and use the general composition rule to determine the convexity.

sufficient, but not necessary for convexity.

- $f(x)=\sqrt{1+x^2}$ is convex, but can't be proved by Constructive Convexity verification.



### Disciplined Convex Program (DCP)

framework for describing convex optimization problems based on constructive convex analysis.

##### Definition

a DCP has

- zero or one objective with form
  - minimize {scalar convex expression} or
  - maximize {scalar concave expression}
- zero or more constraints, with form
  - {convex expression} <= {concave expression} or
  - {concave expression} >= {convex expression} or
  - {affine expression} == {affine expression} 

Expressions are formed from variables, constants, and functions have known convexity, monotonicity and sign properties.

##### Canonicalization

DCP is very easy to build a parser/analyzer, and be transformed to cone form, then solved by some generic solver.

CVXPY will raise error if the constraints not obey the DCP rules.





