# bash



* options
  * `--`: disable further option processing.
  * `-i`: interactive
  * `-l`: login-shell
* `man [command]` : show manual for command 
* `apropos [reg-exp]`: search in the DESCRIPTION field of all manuals.

* shortcuts
  * `ctrl-w` delete last word
  * `ctrl-a/e`: move cursor to the beginning/ending.
  * `ctrk-l`: clear screen
  * `alt-b/f`: move cursor word-wise
  * `alt-#`: comment the current command and run it.
* `!$` the parameters from the last command.
* `pgrep -a [pattern]`: list processes matching pattern. (a means list-all)
  * similar to `ps aux | grep [pattern]`
* `pkill [pattern]`: kill processes matching pattern.

* traceback bash error:

  ```bash
  set -euo pipefail
  trap "echo 'error: Script failed: see failed command above'" ERR
  ```

* locally move working directory:

  ```bash
  # do something in current dir
  (cd /some/other/dir && other-command)
  # continue in original dir
  ```

* login shells v.s. non-login shells

  *  `/etc/profile, ~/.profile, ~/.bash_profile`: for login shells (**the first process that executes under your user ID**, e.g., by `su` or `ssh`)
  * `~/.bashrc`: for interactive non-login shells (shells started in an existing session).

* `sudo -u [user]`: run command as user.

* `repren`: batch rename

  * [tbc]

* the fastest way to delete large amount of files:

  ```bash
  mkdir empty && rsync -r --delete empty/ some-dir && rmdir some-dir
  ```

* `tac`: reversed `cat`

* `ncdu`: n-curses version `du`.

* `tar xf <file>`: one command to uncompress all kind of `tar/tar.xz/tar.gz`