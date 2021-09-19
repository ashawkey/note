# MySQL

### Install

```bash
apt install mysql-server

systemctl start mysql
systemctl status mysql

# config:
vim /etc/mysql/mysql.conf.d/mysqld.cnf
```



### CLI

```bash
# login
mysql -u root -p
# show databases
show databases; # = show schemas;
# select database
use <db>;
# show tables
show tables; # from the current db
show tables from <db>; # from <db>
# describe table
describe <tb>; # = desc <tb>; = explain <tb>; = show columns from <tb>;
# display first <num> rows
select * from <tb> order by <col> [asc | desc] limit <num>;

# show mysql databases location
select @@datadir; # default: /var/lib/mysql

# show size of all dbs
SELECT table_schema AS "Database", SUM(data_length + index_length) / 1024 / 1024 AS "Size (MB)" FROM information_schema.TABLES GROUP BY table_schema

# show full columns desp of table
show full columns from <tb>;

```



### backup & recovery

```bash
# backup
# ref: https://dev.mysql.com/doc/mysql-backup-excerpt/5.7/en/mysqldump-sql-format.html
mysqldump --all-databases > backup.sql
mysqldump -databases <db1> [db2 db3 ...] > backup.sql
mysqldump <db> [table1 table2 ...] > backup.sql

# load
shell $ mysql < backup.sql
mysql $ source backup.sql;
```







### Set password

```mysql
USE mysql;
UPDATE user SET plugin='mysql_native_password' WHERE User='root';
FLUSH PRIVILEGES;
exit;
```

the default password is ''.



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





