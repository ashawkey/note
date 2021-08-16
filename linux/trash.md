# trash-cli

Cut my fingers if I `rm` a file accidentally again!!!



### Usage

```bash
# install
sudo apt install trash-cli

# rm a file or directory (recursive, force)
trash <file/dir>
trash-put <file/dir>

# list the trashbin
trash-list

# empty the trashbin (globally!)
trash-empty

# empty certain file from the trashbin
trash-rm <pattern>

# restore from trash
restore-trash # will show a list and let you choose id to restore
```



### Details

`trash` literally moves the file to `<TrashDir>/files`, and create an info file at `<TrashDir>/infos`.

About the location of `<TrashDir>`:

* If you delete files under your home, the trash directory is `~/.local/share/Trash`

* If you delete files under another disk, like `/data/`, the trash directory is `/data/.Trash`

  ```bash
  # you may get a warning like:
  $ trash-list
  TrashDir skipped because parent not sticky: /data/.Trash/1002
  
  # solution: make it sticky
  $ chmod +t /data/.Trash
  ```
