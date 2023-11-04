# Pandas 

> Credit to `https://github.com/hangsz/pandas-tutorial`

```python
import numpy as np
import pandas as pd
```



### IO

```python
# load
df = pd.read_csv(path, header=None) # no column names (first line is data)

# save
df.to_csv(path, header=False, index=False) # do not write row and column names (will be default int)
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
df = pd.read_csv(filepath_or_buffer, sep=',', header='infer', index_col=None, names=None)
# If csv has no header: set header=None
# index_col=None: create a new range. 
# index_col=0: use the first col 
# index_col='name': use the 'name' col

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

### unique
s.unique()
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
df[0:1] # row, return DF! use df.iloc[0] to get same data as Series
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

### unique
df.drop_duplicates()


### List as column
## select non empty rows
df[df['list'].str.len() > 0]
```



### Conditioned Query

```python
df.loc[df.country == 'Italy']
df.loc[df.country.isin(['Italy', 'New Zealand'])]
df.loc[(df.points>80) & (df.country=='Italy')] # Never use `and`, `or`. Must use ()&(), ()|()
df.loc[df.price.notnull()] # or isnull()
```



### Grouping and Sorting

```python
df.groupby(by=None, axis=0, ...)
# combine rows, followed by a function.

>>> df
   Animal  Max Speed
0  Falcon      380.0
1  Falcon      370.0
2  Parrot       24.0
3  Parrot       26.0

>>> df.groupby(['Animal']).mean()
        Max Speed
Animal
Falcon      375.0
Parrot       25.0

>>> df.groupby(['Animal'])
<pandas.core.groupby.groupby.DataFrameGroupBy object at 0x000002B607F06128>
```





### Missing values

```python
### find na rows

is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
print(rows_with_NaN)

### fillna
def encode_label(train, columns, test=None):
    from sklearn.preprocessing import LabelEncoder
    for column in columns:
        train[column] = train[column].fillna('na')
        enc = LabelEncoder()
        enc.fit(train[column])
        train[column] = enc.transform(train[column])
        if test is not None:
            test[column] = test[column].fillna('na')
            test[column] = enc.transform(test[column])
    return train if test is None else train, test

def impute(train, columns, test=None):
    from sklearn.impute import SimpleImputer
    for column in columns:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train[[column]])
        train[column] = imp.transform(train[[column]])
        if test is not None:
            test[column] = imp.transform(test[[column]])      
    return train if test is None else train, test
```



### Melting

```python
df.melt(id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)
# id_vars[list]: cols to use as identifier variables. If None, use nothing.
# value_vars[list]: cols to unpivot. If None, use all cols not in id_vars.
# var_name[scalar]:  name for 'variable' col.
# value_name[scalar]: name for 'value' col.
>>> df
     Name    Course  Age
0    John   Masters   27
1     Bob  Graduate   23
2  Shiela  Graduate   21

>>> df.melt(id_vars=['Name'])
     Name variable     value
0    John   Course   Masters
1     Bob   Course  Graduate
2  Shiela   Course  Graduate
3    John      Age        27
4     Bob      Age        23
5  Shiela      Age        21

>>> df.melt(id_vars=['Name', 'Course'], value_vars=['Age'])
     Name    Course variable  value
0    John   Masters      Age     27
1     Bob  Graduate      Age     23
2  Shiela  Graduate      Age     21

>>> df.melt(value_vars=['Age'])
  variable  value
0      Age     27
1      Age     23
2      Age     21
```

### Pivoting

```python
df.pivot(index=None, columns=None, values=None)
# index[str/obj]: index for new df. If None, use current.
# columns[str/obj]: col for new df.
# values[str/obj/list]: If None, use all remained.

>>> df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                      'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                      'baz': [1, 2, 3, 4, 5, 6],
                      'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
>>> df
    foo   bar  baz  zoo
0   one   A    1    x
1   one   B    2    y
2   one   C    3    z
3   two   A    4    q
4   two   B    5    w
5   two   C    6    t

>>> df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
      baz       zoo
bar   A  B  C   A  B  C
foo
one   1  2  3   x  y  z
two   4  5  6   q  w  t

>>> df.pivot(index='foo', columns='bar', values='baz')
bar  A   B   C
foo
one  1   2   3
two  4   5   6
```



### Merging

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



### Others

```python
# arithm
df.add(other, axis='columns')
df1.dot(df2)
df.abs()
df.clip(mn, mx)
df.sum(axis='rows') 

# functional
df.apply(func, axis=0, raw=False, ...)
df.pipe(func, *args, **kwargs)
```



