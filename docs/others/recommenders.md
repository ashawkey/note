# Recommender System

### Reference

* https://developers.google.com/machine-learning/recommendation



### Problem Definition

Recommender algorithm can be defined as  a matrix completion problem.

We have $n$ users and $m$ items, then we get a matrix $R _{n \times m}$ with sparse entries $r_{ui}$ as the rating for item $i$ of user $u$, and many missing entries.

The goal is to complete the missing entries to estimate unobserved ratings.

##### CTR (Click-Through Rate) Estimation

an application of recsys. (e.g., in app store)

Each user/item may have additional features to use.





### Large scale Recommendation System pipeline

#### Candidate Generation

Motivation: We don't need to predict a dense $R$, for each user, only a smaller candidate set is needed.

A coarse & fast model is used.

##### Content-based filtering

Uses *similarity between items* to recommend items similar to what the user likes.

Example: If user A watches two cute cat videos, then the system can recommend cute animal videos to that user.

##### Collaborative filtering

Uses *similarities between queries and items simultaneously* to provide recommendations.

Example: If user A is similar to user B, and user B likes video 1, then the system can recommend video 1 to user A (even if user A hasn’t seen any videos similar to video 1).

* ALS is an example of collaborative filtering.
* DL based.

#### Scoring

A precise & slower model is used.

#### Re-ranking

post processing, such as remove explicitly disliked items.







### Alternating Least Square (ALS)

Main idea: Matrix factorization.

Define user matrix $X_{k \times n} = [x_1, x_2, \cdots, x_n]$, item matrix $Y_{k \times m} = [y_1, y_2, \cdots, y_m]$, with $k$ dimensional features. ($k \ll n,m$)

Assume $R \approx X^TY$.

Optimize:


$$

\min_{X,Y} \sum_{r_{ui}}^{\text{observed}} (r_{ui} - x_u^Ty_i)^2 + \lambda (\sum_u||x_u||^2+\sum_i||y_i||^2)

$$


This is nonconvex, but we can make a 2-step iterative optimization to separately optimize X and Y:

* Repeat until Converge: (**WHY???**)

  * Fix Y, update X

    
$$

    x_u = (\sum_{r_{ui} \in r_{u*}}y_iy_i^T+\lambda I_k)^{-1} \sum_{r_{ui} \in r_{u*}}r_{ui}y_i
    
$$


  * Fix X, update Y

    
$$

    y_i = (\sum_{r_{ui} \in r_{*i}}x_ux_u^T+\lambda I_k)^{-1} \sum_{r_{ui} \in r_{*i}}r_{ui}x_u
    
$$


* Inference:

  
$$
 
  r_{ui} = x_u^Ty_i
  
$$






### Wide & Deep (2016, Google)

wide = Linear Regression (Memorization)

deep = MLP (Generalization)

