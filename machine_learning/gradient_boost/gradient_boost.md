# Gradient Boost

### LightGBM

```python
import lightgbm as lgb

train_data = lgb.Dataset(train[features], label=train['label'])
valid_data = lgb.Dataset(valid[features], label=valid['label'])
test_data = lgb.Dataset(test[features], label=test['label'])

params = {
    'num_classes': 64,
    'objective': 'binary',
    'metric': 'auc',
}
num_round = 1000

### train
best = lgb.train(params, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10)

### test
from sklearn import metrics
ypred = best.predict(test[features])
score = metrics.roc_auc_score(test['label'], ypred)
```

