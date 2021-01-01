# Feature Engineering





```python
### Label Encoding
from sklearn import preprocessing

cat_features = ['app', ...]
le = preprocessing.LabelEncoder()
for feature in cat_features:
    df[feature+"_label"] = le.fit_transform(df[feature])
```

