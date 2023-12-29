# Statistical test in python

```python
from scipy.stats import ttest_ind, ttest_rel, normaltest, ks_2samp

## Is x from normal distribution ?
v, p = normaltest(x)
if p >= 0.05:
    print("x is normal.")
else:
    print("x is not normal.")
    
    
## Compare Mean of x and y
# if x and y are normal, and have same variance (Student's t-test)
v, p = ttest_ind(x, y, equal_val=True)
if p >= 0.05:
    print("Not significant difference.")
else:
    print("Significant difference.")
    
# if x and y are normal, and have different variance (Welch's t-test)
v, p = ttest_ind(x, y, equal_val=False)
if p >= 0.05:
    print("Not significant difference.")
else:
    print("Significant difference.")
    
# if x or y is not normal (Kolmogorov–Smirnov test)
v, p = ks_2samp(x, y)
if p >= 0.05:
    print("Not significant difference.")
else:
    print("Significant difference.")
    
# if x and y are paired (Paired t-test)
v, p = ttest_rel(x, y)
if p >= 0.05:
    print("Not significant difference.")
else:
    print("Significant difference.")


```


