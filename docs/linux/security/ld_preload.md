# LD_PRELOAD

To fix `[ERROR: ld.so: object LD_PRELOAD cannot be preloaded: ignored]`

```bash
# this file is hidden, so use `ls -a`
sudo echo "" > /etc/ld.so.preload
```

