# MySQL

### Login

```bash
 mysql -u root -p
```



### Data types

* TINYINT, SMALLINT, MEDIUMINT, INT, BIGINT
* DEMICAL, FLOAT, DOUBLE, BIT

* CHAR(len)

  fixed-length string

* VARCHAR(max-len)

  variable-length string (<=65535)

* BINARY, VARBINARY

* TINYBLOB, BLOB, MEDIUMBLOB, LONGBLOB

* TINYTEXT, TEXT, MEDIUMTEXT, LONGTEXT

  variable-length string. (at most 255, 65535, 16777215, 4294967295 characters)

* DATE, TIME, DATETIME, TIMESTAMP



### Create database

You must create database first, then select database, then create table.

```mysql
CREATE DATABASE [IF NOT EXISTS] database_name;
[CHARACTER SET charset_name]
[COLLATE collation_name]

USE database_name;
```



### Create table

```mysql
CREATE TABLE [IF NOT EXISTS] table_name(
   column_1_definition,
   column_2_definition,
   ...,
   table_constraints
) ENGINE=storage_engine;

-- column definition
column_name data_type(length) [NOT NULL] [DEFAULT value] [AUTO_INCREMENT] column_constraint;
```



### Operations

Nearly the same as SQLite.



### Full text search

* Create index by `FULLTEXT (columns) WITH PARSER ngram`

  ```mysql
  CREATE TABLE articles (
      id INT UNSIGNED AUTO_INCREMENT NOT NULL PRIMARY KEY,
      title VARCHAR (200),
      body TEXT,
      FULLTEXT (title, body) WITH PARSER ngram
  ) ENGINE = INNODB;
  ```

* Search `WHERE MATCH (columns) AGAINST (pattern)`

  * Default: `IN NATURAL LANGUAGE MODE`
  * Bool operation: `IN BOOLEAN MODE`

  ```mysql
  SELECT * FROM articles
  WHERE MATCH (title,body) -- must be same as FULLTEXT
  AGAINST ('一路 一带' IN NATURAL LANGUAGE MODE);
  
  // 不指定模式，默认使用自然语言模式
  SELECT * FROM articles
  WHERE MATCH (title,body)
  AGAINST ('一路 一带');
  
  // 必须包含"腾讯"，但是不能包含"通讯工具"
  SELECT * FROM articles
  WHERE MATCH (title,body)
  AGAINST ('+腾讯 -通讯工具' IN BOOLEAN MODE);
  ```



### Python API

```python
import mysql.connector

# the same as sqlite3

conn = mysql.connector.connect(user='root', password='password', database='test')
cursor = conn.cursor()
cursor.execute().fetchone() # fetchall()
cursor.close()
conn.commit()
conn.close()
```

Flask-MySQL:

```python
from flaskext.mysql import MySQL

mysql = MySQL()
mysql.init_app(app)

cursor = mysql.get_db().cursor()

# insert
stmt = "insert into table(columns) values(%s, %s)"
cursor.execute(stmt, (val1, val2))

# select
stmt = "select * from table"
cursor.execute(stmt)
res = cursor.fetchall() # list of list
```





