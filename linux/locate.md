# locate

```bash
# find file by name
locate <pattern>
```

This command is paired with `updatedb`, which creates a database for `locate` for fast filename search.

`updatedb` is run by `cron` everyday.



This command is usually used to locate libraries.

```bash
locate libglog
locate opencv

```

