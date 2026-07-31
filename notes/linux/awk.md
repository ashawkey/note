# AWK 

```bash
# reserved name
$0: the whole line
$1, $2, ... : cells in a line
FS: field separator, defaults to ' '
RS: record separator, defaults to '\n'
NF: number of field
NR: number of record
OFS: output field separator, defaults to ' '
ORS: output record separator, defaults to '\n'

# FS
awk -F '[[:space:]+]' # default
awk -F ':' # :+ (one or more)
awk -F '[ ,]+'

# print
awk '{print "a" "b" "c" $1}'
awk '{print $3" "$7}'

# if
awk '{if(NR>=20 && NR<=30){print $1}}'

# operator
# same as C/C++

# regex
awk 'if($0~/pattern/){}'
awk '/pattern/{}' # ditto
awk 'if($0 !~ /pattern/){}'


# bool
awk '($1=="root"){}'

# variable
# inited to 0
awk 'BEGIN{print a++,++a}' # 0 2
awk 'BEGIN{a="20b4";print a++,++a}' # 20 22

# BEGIN END
awk 'BEGIN{count=0;} {count++;} END{print count;}'

# code file
awk -f code.txt data.txt

## Applications
# paragraph
awk 'BEGIN{FS="\n";RS=""}'
# mean number of column
awk '{total += $1; count++} END {print total/count}' log.txt

```

### References

![](https://images2015.cnblogs.com/blog/1089507/201701/1089507-20170126232437800-1355193233.jpg)