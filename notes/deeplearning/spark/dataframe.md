# Spark SQL


```python
from pyspark.sql import SparkSession

# create spark
spark = SparkSession.builder.getOrCreate()
    
# load data
df = spark.read.json('test.json') # json
df = spark.read.csv('foo.csv', header=True) # csv

# create by Row
from pyspark.sql import Row
df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])

# create from explicit schema
df = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')

# create from pandas df
pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df = spark.createDataFrame(pandas_df)

# view data
df.columns # cols
df.printSchema() # detailed cols definition
df.show() # all cols, default first 20 rows
df.show(1) # all cols, the first row
df.select('name').show() # specific col
df.filter(df['age'] > 21).show() # filter by col condition
df.groupBy("age").count().show() # group

# sql query
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()

# write / export data
pandas_df = df.toPandas() # pandas
df.write.csv('foo.csv', header=True) # csv


```

