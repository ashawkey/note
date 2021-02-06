### ps

```bash
# most usual usecase:
ps -aux | grep <pattern>
```



### kill

kill by pid. single wrapper to kill system call.

```bash
kill [-<signal>] <pid>

# equivalent:
kill -9 <pid>
kill -KILL <pid>
kill -SIGKILL <pid>

# 1: HUP, hang up or restart.
# 2: INT, Ctrl+C
# 9: KILL
# 15: TERM, default
# 19: STOP, Ctrl+Z
```



### killall

kill by name.

```bash
killall [-<signal>] <name>

# kill all processes of a user
killall -user <user>

# case in-sensitive
killall -I <name>
```



### pkill

kill by name, use `pgrep` for matching.

```bash
pkill [-<signal>] <pattern>

# kill root's command that match name
pkill -u root name

# kill all processes of a user
pkill -u <user>
```

