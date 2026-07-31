# vim


### remove ^M at line ending (dos to unix format)

```
:e ++ff=dos 
```


### slow startup

First, try `vim -X`.

If it solves the problem, add this to `~/.vimrc`:

```
set clipboard=exclude:.*
```

This tells vim do not connect to the X server for clipboard sharing.
