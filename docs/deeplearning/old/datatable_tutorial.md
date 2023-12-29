# datatable

```python
import datatable as dt
```


### Load

```python
dt.Frame(A=range(5), B=['a','b','c','d','e'])
dt.Frame({"A": [1,2], "B": ['a', 'b']})

dt.Frame(pandas_dataframe)
dt.Frame(numpy_array)

dt.fread("test.csv")
dt.fread(data, sep=None, header=None, fill=False, skip_blank_lines=False, columns=None)
```


### Properties

```python
DT.shape
DT.names
DT.stypes # column types
```


### Data Manipulation


$$
\displaylines{
DT[i, j, by(), sort(), join()]
}
$$


```python
# selector
DT[i, j] = 1 # [row, col, ...]
del DT[i, j] 

# frame proxy
from datatable import f, max, min, sum
DT[:, (f.A-min(f.A))/(max(f.A)-min(f.A))]

f.A
f['A']
f[0]
f[:]          # select all columns
f[::-1]       # select all columns in reverse order
f[:5]         # select the first 5 columns
f[3:4]        # select the fourth column
f["B":"H"]    # select columns from B to H, inclusive
f[int]        # select all integer columns
f[float]      # select all floating-point columns
f[dt.str32]   # select all columns with stype `str32`
f[None]       # select no columns (empty columnset)

f[int].extend(f[float])          # integer and floating-point columns
f[:3].extend(f[-3:])             # the first and the last 3 columns
f.A.extend(f.B)                  # columns "A" and "B"
f[:].extend({"cost": f.price * f.quantity}) # add new column
f[:].remove(f[str])    # all columns except columns of type string
f[:10].remove(f.A)     # the first 10 columns without column "A"
f[:].remove(f[3:-3])   # same as `f[:3].extend(f[-3:])`
                       
    
DT[:, "A"]         # select 1 column
DT[:10, :]         # first 10 rows
DT[::-1, "A":"D"]  # reverse rows order, columns from A to D
DT[27, 3]          # single element in row 27, column 3 (0-based)
DT[(f.x > mean(f.y) + 2.5 * sd(f.y)) | (f.x < -mean(f.y) - sd(f.y)), :]
del DT[:, "D"]     # delete column D
del DT[f.A < 0, :] # delete rows where column A has negative values

# compute new column
DT[:, {"x": f.x, "y": f.y, "x+y": f.x + f.y, "x-y": f.x - f.y}]

# append
DT1.cbind(DT2, DT3)
DT1.rbind(DT4, force=True)

# aggeregate with by
DT[:, sum(f.quantity), by(f.product_id)]

# join (left outer join)
DT[:, sum(f.quantity * g.price), join(products)]

# sort
DT.sort("A")
DT[:, :, sort(f.A)]
```


### Save

```python
DT.to_pandas()
DT.to_numpy()
DT.to_dict()
DT.to_list()

DT.to_csv("out.csv")
DT.to_jay("out.jay") # binary
```

