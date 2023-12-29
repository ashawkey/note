# cadaver

### WebDAV (Web Distributed Authoring and Versioning)

an extension of the [Hypertext Transfer Protocol](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol) (HTTP) that allows [clients](https://en.wikipedia.org/wiki/Web_client) to perform remote [Web](https://en.wikipedia.org/wiki/World_Wide_Web) content authoring operations.


### cadaver

A command-line WebDAV client for Unix.

Use `cadaver` to backup data:

* Setup WebDAV key

  use `jianguoyun` to register an application key.

* install `cadaver`

  ```
  sudo apt install cadaver
  ```

* configurations for `no-passwd` login

  edit `~/.netrc`

  ```
  machine dav.jianguoyun.com
  login <email>
  password <passwd>
  ```

* automatic script 

  edit `/root/backup/backup` and `chmod 777`

  ```bash
  #! /bin/bash
  
  mysqldump db > /root/backup/db.sql
  cadaver https://dav.jianguoyun.com/dav/ < /root/backup/backup_command.txt
  ```

  edit `/root/backup/backup_command.txt`

  ```
  cd hawia
  put /root/backup/db.sql
  bye
  ```

* setup crontab

  ```bash
  cp /root/backup/backup /etc/cron.daily/
  ```

  