# gdb

## Tutorial

https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/gdb.html


## Deal with Segmentation Fault

Set `-g` flag for compiler,

```bash
g++ -g -Wall -O3 ...
```

Set core dump,

```bash
ulimit -c unlimited # this should be run every time 
```

Run program and got `Segmentation Fault (core dumped)`,

```bash
./god_damn_program 
# Segmentation Fault
# a `core` should be generated in current dir
```

Run gdb,

```bash
gdb ./god_damn_program ./core

$gdb info stack # show traceback
# ctrl-d to exit
```


### seg fault in python that calls c

```bash
gdb --args python main.py
....................
......greetings.....
....................
run
....................
... the seg fault...
....................
bt
```

