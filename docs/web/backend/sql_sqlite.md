# Sqlite3

> SQL: Structured Query Language

### Basics

* `.sql` file

  contains a series of `sql` script.


### Data Types

* NULL
* INTEGER
* REAL
* TEXT: character data, unlimited size.
* BLOB: binary large object, can store any time of data. unlimited size.


### Create Table

```sqlite
CREATE TABLE [IF NOT EXISTS] [schema_name].table_name (
	column_1 data_type PRIMARY KEY,
   	column_2 data_type NOT NULL,
	column_3 data_type DEFAULT 0,
	table_constraints
) [WITHOUT ROWID];
```

* `primary key`: must be unique.

* `rowid`: by default, SQLite adds an implicit column `rowid interger` . 

  It is used to organize data as a B-tree.

* `foreign key`:

  ```sqlite
  -- parent table
  CREATE TABLE supplier_groups (
  	group_id integer PRIMARY KEY,
  	group_name text NOT NULL
  );
  
  -- child table
  CREATE TABLE suppliers (
      supplier_id   INTEGER PRIMARY KEY,
      supplier_name TEXT    NOT NULL,
      group_id      INTEGER NOT NULL, -- define foreign key first
      FOREIGN KEY (group_id) REFERENCES supplier_groups (group_id)
  );
  ```

  this makes it impossible to insert `supplier` with non-existing `group_id` in `supplier_groups`.

  when deleting & updating in `supplier_groups`:

  ```sqlite
  FOREIGN KEY (foreign_key_columns)
     REFERENCES parent_table(parent_key_columns)
        ON UPDATE action 
        ON DELETE action;
  ```

  Actions:

  * no action [Default !]

  * set null
  * set default
  * restrict (Cannot delete `supplier_groups` key unless all `supplier` with this key has been deleted.)
  * cascade (Delete all `supplier` with this key too)

### Operations

* INSERT

  ```sqlite
  INSERT INTO table (column1,column2 ,..)
  VALUES(value1,	value2 ,...);
  VALUES(value1,	value2 ,...); -- second line
  ...
  ```

  

* DELETE

  ```sqlite
  DELETE FROM table
  WHERE search_condition;
  ```

  

* SELECT

  ```sqlite
  SELECT	1 + 1;
  
  SELECT column1, column2 FROM table;
  SELECT * FROM table;
  
  -- full syntax
  SELECT DISTINCT column_list
  FROM table_list
    JOIN table ON join_condition
  WHERE row_filter
  ORDER BY column
  LIMIT count OFFSET offset
  GROUP BY column
  HAVING group_filter;
  ```

  * DISTINCT

    remove duplicated results.

    ```sqlite
    SELECT DISTINCT city FROM customers ORDER BY city;
    ```

  * WHERE

    ```sqlite
    WHERE column_1 = 100;
    WHERE column_2 IN (1,2,3);
    WHERE column_3 LIKE '%pattern%';
    WHERE column_4 BETWEEN 10 AND 20;
    
    WHERE albumid = 1 AND milliseconds > 250000; -- logical operation
    ```

    > LIKE pattern
    >
    > %: * (any length wildcard)
    >
    > _: . (single char wildcard)

  * ORDER BY

    order by which column

  * LIMIT row_count OFFSET offset

    constrain the max number of rows returned.

    count starting from OFFSET.

    


* UPDATE

  used to change value of inserted row.

  ```sqlite
  UPDATE table
  SET column_1 = new_value_1,
      column_2 = new_value_2
  WHERE
      search_condition 
  ORDER column_or_expression
  LIMIT row_count OFFSET offset;
  ```

  


### PRAGMA

Set default behavior.

```sqlite
PRAGMA case_sensitive_like = true;
```


### Full text search

We must use FTS5 virtual table to achieve full text search in SQLite.

不支持中文搜索，需要另外的程序切词！

```sql
create virtual table posts using fts5(title, ctime, mtime, body);

SELECT * 
FROM posts 
WHERE posts MATCH 'fts5'; /* By default, search is case-independent. */

-- equals

SELECT * 
FROM posts 
WHERE posts = 'fts5';

-- equals

SELECT * 
FROM posts('fts5');
```

Highlight Aux:

```sql
SELECT highlight(posts,0, '<b>', '</b>') title, 
       highlight(posts,1, '<b>', '</b>') body
FROM posts 
WHERE posts MATCH 'SQLite'
ORDER BY rank;
```


### Python API

```python
import sqlite3

# connect
conn = sqlite3.connect('example.db')
c = conn.cursor()

# operations
# This is the qmark style:
c.execute("insert into people values (?, ?)", (who, age))
# And this is the named style:
c.execute("select * from people where name_last=:who and age=:age", {"who": who, "age": age})
print(c.fetchone())

# save & close
conn.commit()
conn.close()
```

* Operations:

  ```python
  # executemany
  purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
               ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
               ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
              ]
  c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
  
  # executscript
  c.executescript("""
      create table book(
          title,
          author,
          published
      );
  
      insert into book(title, author, published)
      values (
          'Dirk Gently''s Holistic Detective Agency',
          'Douglas Adams',
          1987
      );
      """)
  
  c.fetchone()
  # Fetches the next row of a query result set, returning a single sequence, or None when no more data is available.
  
  c.fetchall()
  # Fetches all (remaining) rows of a query result, returning a list. Note that the cursor’s arraysize attribute can affect the performance of this operation. An empty list is returned when no rows are available.
  ```

  

* `conn.row_factory`

  By default, `cur.fetchone()` returns a `tuple`.

  This is in fact defined by the `row_factory` attribute.

  ```python
  def dict_factory(cursor, row):
      d = {}
      for idx, col in enumerate(cursor.description):
          d[col[0]] = row[idx]
      return d
  
  con = sqlite3.connect(":memory:")
  con.row_factory = dict_factory
  cur = con.cursor()
  cur.execute("select 1 as a")
  cur.fetchone() # a dict now !
  ```

* `Row`

  A highly optimized `row_factory`.

  ```python
  conn.row_factory = sqlite3.Row
  c = conn.cursor()
  c = execute('select * from stocks')
  r = c.fetchone()
  
  tuple(r) # default tuple
  r.keys() # keys
  len(r) # length
  r[key] # value
  
  ```

  