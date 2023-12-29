### chattr

https://www.computerhope.com/unix/chattr.htm#list-of-attributes

* `lsattr <file>`

  list file attribute

  ```bash
  # output example
  ----i-------- /etc/resolv.conf
  
  # some attributes:
  a: append mode only for writing
  i: immutable (cannot be deleted or renamed, `sudo rm -f` will show `Operation not permitted`, but not `Permission denied` )
  ```

  

* `chattr [-R] [+/-/=<attr>] <file>`

  change the attributes of file or directory.


### chown

change ownership of file or directory.

**only root can use chown.**

similarly, there is `chgrp` for change group of file or directory.

```bash
sudo chown <new-owner> <file>
sudo chown <new-owner:new-group> <file>
sudo chown -R <new-owner> <dir>
```


### umask

set/check default user permission mask for new files.

```bash
umask # show current default umask
umask 022 # rw-r--r--

# u[ser], g[roup], o[ther], a[ll]
umask a= # dangerous! ---------
umask u+x # rwxr--r--
```


### chmod

set user permission mask for a specific file.

```bash
chmod 777 <file> # rwxrwxrwx
chmod u=rwx,g=rx,o=r <file>

chmod -R 777 <dir>
```


### note the escaped space!

```bash
`python\ ` looks the same as `python ` in default ls and top.
but `ls --escape` will reveal it!
```


