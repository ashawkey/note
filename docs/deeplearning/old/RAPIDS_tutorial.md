# RAPIDS

### cudf

GPU DataFrame.


### cuml

Machine Learning Algorithms.

##### k-NN

* find duplicates from image embedding

  ```python
  import cuml
  
  model = cuml.neighbors.NearestNeighbors(n_neighbors=3)
  model.fit(embed_train)
  distances, indices = model.kneighbors(embed_test)
  
  mm = np.min(distances, axis=1)
  
  idx = np.where((mm < 2))[0]
  ```


##### k-Means

```python
import cuml

model = cuml.KMeans(n_clusters=20)
model.fit(embed)
train['cluster'] = model.labels_
train.head()
```


##### T-SNE

```python
import cuml

# model
model = cuml.TSNE()
embed2D = model.fit_transform(embed)
train['x'] = embed2D[:,0]
train['y'] = embed2D[:,1]

# get region with largest MM rate
X_DIV = 10; Y_DIV = 10
x_min = train.x.min()
x_max = train.x.max()
y_min = train.y.min()
y_max = train.y.max()
x_step = (x_max - x_min)/X_DIV
y_step = (y_max - y_min)/Y_DIV
mx = 0; xa_mx = 0; xb_mx=0; ya_mx = 0; yb_mx = 0
for k in range(X_DIV+1):
    for j in range(Y_DIV+1):
        xa = k*x_step + x_min
        xb = (k+1)*x_step + x_min
        ya = j*y_step + y_min
        yb = (j+1)*y_step + y_min
        df = train.loc[(train.x>xa)&(train.x<xb)&(train.y>ya)&(train.y<yb)]
        t = df.target.mean()
        if (t>mx)&(len(df)>=16):
            mx = t
            xa_mx = xa
            xb_mx = xb
            ya_mx = ya
            yb_mx = yb
        #print(k,j,t)
        
# vis
plt.figure(figsize=(10,10))
df1 = train.loc[train.target==0]
plt.scatter(df1.x,df1.y,color='orange',s=10,label='Benign')
df2 = train.loc[train.target==1]
plt.scatter(df2.x,df2.y,color='blue',s=10,label='Malignant')
plt.plot([xa_mx,xa_mx],[ya_mx,yb_mx],color='black')
plt.plot([xa_mx,xb_mx],[ya_mx,ya_mx],color='black')
plt.plot([xb_mx,xb_mx],[ya_mx,yb_mx],color='black')
plt.plot([xa_mx,xb_mx],[yb_mx,yb_mx],color='black')
plt.legend()
plt.show()
```

