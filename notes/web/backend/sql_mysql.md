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


### Char set

> ref: https://mathiasbynens.be/notes/mysql-utf8mb4#character-sets

To enable full support for `utf` charset, we need to change the default settings from `utf8` to `utf8mb4`.

* `utf8`: short for `utf8mb3`, which means `max bytes 3`, so it can use at most 3 bytes to store chars.
* `utf8mb4`: max bytes 4, this supports all the strange chars, such as :flags:.

> MySQL’s `utf8` encoding is awkwardly named, as it’s different from proper UTF-8 encoding. It doesn’t offer full Unicode support, which can lead to data loss or security vulnerabilities.

To set the default charset to `utf8mb4`:

* Alter the databases have created:

  ```sql
  # For each database:
  ALTER DATABASE database_name CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;
  
  # For each table:
  ALTER TABLE table_name CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  
  # For each column (example):
  ALTER TABLE table_name CHANGE column_name column_name VARCHAR(191) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  ```

* modify default configs at `/etc/mysql/my.cfg`

  ```
  [client]
  default-character-set = utf8mb4
  
  [mysql]
  default-character-set = utf8mb4
  
  [mysqld]
  character-set-client-handshake = FALSE
  character-set-server = utf8mb4
  collation-server = utf8mb4_unicode_ci
  ```

  And restart `mysql` service.

  Check by:

  ```mysql
  mysql> SHOW VARIABLES WHERE Variable_name LIKE 'character\_set\_%' OR Variable_name LIKE 'collation%';
  +--------------------------+--------------------+
  | Variable_name            | Value              |
  +--------------------------+--------------------+
  | character_set_client     | utf8mb4            |
  | character_set_connection | utf8mb4            |
  | character_set_database   | utf8mb4            |
  | character_set_filesystem | binary             |
  | character_set_results    | utf8mb4            |
  | character_set_server     | utf8mb4            |
  | character_set_system     | utf8               |
  | collation_connection     | utf8mb4_unicode_ci |
  | collation_database       | utf8mb4_unicode_ci |
  | collation_server         | utf8mb4_unicode_ci |
  +--------------------------+--------------------+
  ```

  Finally, you should repair and optimize current tables:

  ```bash
  # one line for all dbs
  mysqlcheck -u root -p --auto-repair --optimize --all-databases
  ```

  (or in a per-table form)

  ```mysql
  # For each table
  REPAIR TABLE table_name;
  OPTIMIZE TABLE table_name;
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


