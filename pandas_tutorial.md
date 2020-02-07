# Pandas 

> Credit to `https://github.com/hangsz/pandas-tutorial`

```python
import numpy as np
import pandas as pd
```



### Data Structure

##### Series (Ordered, Indexed, Homogenous, One-Dimensional)

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=False)
# index := range(0, data.shape[0])
# dtype := float64

s = pd.Series([1,2,3], ['a','b','c'])
s = pd.Series({'a':1, 'b':2, 'c':3}) # indexed data

## attributes
s.name
s.values
s.index
s.dtype
```

##### DataFrame (Indexed, Heterogenous, Two-Dimensional)

Series of Series.

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# index := range(0, data.shape[0])
# columns := range(0, data.shape[1])

data = [[1,2,3],
        [4,5,6]]
index = ['a','b']
columns = ['A','B','C']
df = pd.DataFrame(data=data, index = index, columns = columns)

## attributes
df.index
df.columns
df.values
df.dtypes # of every column
df.shape
df.size
df.head()
df.tail()
df.memory_usage(deep=False) # deep: whether count reference
df.descrive(include='all') # simple stats of columns

## create from file
df = pd.read_csv(filepath_or_buffer, sep=',', header='infer', names=None)
# If csv has no header: set header=None

df = pd.read_excel(io, sheetname=0, header=0)
```



### Manipulate Data

##### Series

```python
### Access
mask = [True,True,False]
## [], index
s[0]
s[0:2]
s[[0,1]]
s[mask]
## .loc[] == []
s.loc['a']
## .iloc[], position
s.iloc[0]
s.iloc[0:2]
s.iloc[[0,2]]
s.iloc[mask]

### Modify value
s[0] = 1
s.replace(to_replace, value, inplace=False) # can replace a list of values

### Modify index
s.index = new_index
s.rename(index=None, level=None, inplace=False)

### append
s.loc[100] = 100
# `s[100] = 100` lead to error, use [] only to access.
s.append(s2, ignore_index=False) # ignore_index: erase index of s2

### delete
del s[100]
s.drop([100, 'a'])
```



##### DataFrame

```python
data = [[1,2,3],
        [4,5,6]]
index = ['a','b']
columns = ['A','B','C']
df = pd.DataFrame(data=data, index=index, columns = columns)

### access
## []
df['A'] # column, return Series
df[['A', 'B']] # multiple columns, return DF
# `df[0]` error, 0 is not a column name.
df[0:1] # row, return DF
df[mask] # row, return DF

## .loc()
df.loc['b'] # == df.loc['b', :], return Series
# `df.loc['B']` error, 'B' not in index (rows)
df.loc[:, 'B'] # return Series
df.loc['b', 'B'] # return scalar
df.loc[['a', 'b'], 'B']
df.loc[mask, 'B']

## .iloc()
df.iloc[0,0]
df.iloc[0:2, 0]
df.iloc[[0,1], [0,2]] # return DF
df.iloc[mask1, mask2]

### modify value
df.loc['a', 'A'] = 0

### modify name
df.index = [...]
df.columns = [...]

### append rows
df.loc['c'] = [7,8,9]
pd.concat([df1, df2], axis=0)

### append cols
df['D'] = [10, 11]
pd.concat([df1, df2], axis=1)

### remove
df.drop(['a', 'b'], axis=0) # row
df.drop(['A', 'B'], axis=1) # col
```



### Merge

```python
pd.merge(left, right, how='inner', on=None)
# how: inner, outer, left, right
```



### Options

```python
pd.get_option(key)
pd.set_option(key, value)
pd.reset_option(key)

"display.max_rows"
"display.max_cols"
"display.precision"
"display.max_colwidth"

```



### Arithmetics

```python
df.add(other, axis='columns')
df1.dot(df2)
df.abs()
df.clip(mn, mx)
df.sum(axis='rows') 

```

