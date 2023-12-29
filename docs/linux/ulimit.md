# ulimit

set user limits. 

The most common case is "Too Many Open Files" Error.

### temporary fix

```bash
# check all limits
ulimit -a

# check number of files (nofile)
ulimit -n # default to 1024, meaning you can at most open 1024 files.

# set
ulimit -n 100000
```


### permanent fix

Modify `/etc/security/limits.conf`, and add:

```bash
* soft nofile 100000
* hard nofile 100000
```


