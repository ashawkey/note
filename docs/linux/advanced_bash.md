### brace expansion

https://www.gnu.org/software/bash/manual/html_node/Brace-Expansion.html

```bash
echo a{b,c}d
# abd acd

mkdir tmp/{a,b}/c
# tmp/a/c, tmp/b/c
```



### run & interrupt parallel commands

https://stackoverflow.com/questions/3004811/how-do-you-run-multiple-programs-in-parallel-from-a-bash-script

```bash
# start three progx in parallel, and make sure these can be killed in one ctrl-c
(trap 'kill 0' SIGINT; 
prog1 & 
prog2 & 
prog3
)
```

`kill 0`:  a special command to kill all processes in the current group (usually this means all programs started in the current shell).

The `trap` command will wait until caught the signal `SIGING` and then call the command `kill 0`, another example:

```bash
# Run something important, no Ctrl-C allowed.
# "" means run nothing, thus ignore the signal.
trap "" SIGINT
important_command

# Less important stuff from here on out, Ctrl-C allowed.
# - is used to reset the default behaviour
trap - SIGINT
not_so_important_command
```

