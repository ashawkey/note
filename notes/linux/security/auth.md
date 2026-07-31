# authorization


### last logins

```bash
# last (last logins)
# it uses /var/log/wtmp
last [-num] [user]
last reboot # show latest reboots

# lastb (last bad, failed logins)
lastb [-num] [user]

# lastlog (report most recent login of all users)
# it uses /var/log/lastlog
lastlog [-u <user>]

# who (list current logged in users)
# it uses /var/log/utmp
who
```


### auth.log

`/var/log/auth.log`

all the authorization logs.

this log is maintained by `rsyslog`

```bash
# check the log
sudo less /var/log/auth.log

# something like lastb
sudo cat /var/log/auth.log | grep "Failed password"
```


If `rsyslog` is working but `auth.log` still failed to show any logs, this maybe caused by ownership of `auth.log` (must not be `root` but `syslog:adm`). Try:

```bash
sudo chown syslog:adm /var/log/auth.log
```


### syslog

`/var/log/syslog`

this log is also maintained by `rsyslog`. 

It contains more information that covers everything.

