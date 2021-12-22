# rclone



### install

```bash
curl https://rclone.org/install.sh | sudo bash
```



### set up remotes

follow the instructions:

```bash
rclone config
```

For example,  we setup a webdav called `nuts`:

```bash
--------------------
[nuts]
type = webdav
url = https://dav.jianguoyun.com/dav/
vendor = other
user = ashawkey1999@gmail.com
pass = *** ENCRYPTED ***
--------------------
```

show all remotes:

```bash
# of course, in config
rclone config
# or just list
rclone listremotes
```

check remote disk:

```bash
rclone about nuts:
```



### usage

* list

  ```bash
  # ls remote:path 
  # ONLY list files (size and path), by default recursive! 
  rclone ls nuts:/hawia
  rclone ls nuts:hawia # same, default remote directory is /
  rclone ls --max-depth 1 nuts:hawia # limit recurse level
  rclone --max-depth 1 ls nuts:hawia # same, flag can be anywhere
  rclone ls nuts:hawia --max-depth 1 # same
  
  # lsl remote:path
  # ONLY list files (size, path, modtime), not recursive
  rclone lsl nuts:hawia
  rclone lsl -R nuts:hawia # recursive
  
  # lsd remote:path
  # ONLY list directories (modtime, path)
  rclone lsd nuts:
  rclone lsd -R nuts: # recursive
  
  # lsf remote:path
  # list both files and directories (path)
  rclone lsf nuts:
  rclone lsf -R nuts: # recursive
  ```

* copy

  ```bash
  # copy source:sourcepath dest:destpath
  # ignore identical files, never delete remote files, automatically create dest directory if not exist.
  
  # only copy the CONTENT, not the directory!
  rclone copy ~/data/ nuts:hawia/data/ # cp ~/data/* nuts:/hawia/data/
  rclone copy ~/data nuts:hawia/data # same
  rclone copy -P ~/data nuts:hawia/data # show progress
  rclone copy -i ~/data nuts:hawia/data # dry run (interactive)
  rclone copy --dry-run ~/data nuts:hawia/data # dry run
  
  # copy from remote to local
  rclone copy nuts:hawia/data ~/data -P
  
  # only copy files changed in recent 24h, without travelling the whole directory (efficient!)
  rclone copy --max-age 24h --no-traverse /path/to/src remote:
  
  # copy single file
  rclone copy ~/test.jpg nuts:hawia/data # cp ~/test.jpg nuts:/hawia/data/
  rclone copy ~/test.jpg nuts:hawia/test.jpg # cp ~/test.jpg nuts:/hawia/test.jpg/ (WRONG USE, it create a folder called test.jpg and then copy the real image to the folder)
  rclone copyto ~/test.jpg nuts:hawia/test.jpg # cp ~/test.jpg nuts:/hawia/test.jpg (CORRECT USE)
  
  rclone copy nuts:hawia/data/test.jpg data # mkdir -p data && cp nuts:hawia/data/test.jpg data/
  rclone copy nuts:hawia/data/test.jpg . # will do nothing, WRONG USE!
  rclone copyto nuts:hawia/data/test.jpg test.jpg # cp nuts:hawia/data/test.jpg test.jpg (CORRECT USE)
  ```

* sync

  ```bash
  # sync SOURCE remote:DESTINATION
  # make remote identical to local
  # ignore identical files, will delete remote files.
  
  rclone sync ~/data nuts:hawia/data
  rclone sync -i ~/data nuts:hawia/data # dry run
  rclone sync -P ~/data nuts:hawia/data # show progress
  ```

* delete

  ```bash
  rclone delete remote:path # only delete files
  rclone delete --rmdirs remote:path # also delete empty dirs
  rclone delete --dry-run remote:path # dry run
  
  # delete file pattern
  rclone delete remote:*.txt # WRONG USE, 400 bad request
  rclone delete remote: --include=*.txt 
  
  # delete all files smaller thatn 100M
  rclone --min-size 100M lsl remote:path
  rclone --min-size 100M delete remote:path
  
  # purge: remove the path with all its content
  rclone purge remote:path
  rclone purge -i remote:path # always dry run !
  ```

* cat

  ```bash
  rclone cat remote:path/to/file
  rclone --include "*.txt" cat remote:path/to/dir
  ```

* mkdir

  ```bash
  rclone mkdir remote:path
  ```

* filtering

  the filtering pattern follows a glob style.

  ```bash
  # basic
  *         matches any sequence of non-separator (/) characters
  **        matches any sequence of characters including / separators
  ?         matches any single non-separator (/) character
  [[!]{character-range}]
            character class (must be non-empty)
  { pattern-list }
            pattern alternatives
  c         matches character c (c != *, **, ?, \, [, {, })
  \c        matches reserved character c (c = *, **, ?, \, [, {, })
  
  # char sets
  Named character classes (e.g. [\d], [^\d], [\D], [^\D])
  Perl character classes (e.g. \s, \S, \w, \W)
  ASCII character classes (e.g. [[:alnum:]], [[:alpha:]], [[:punct:]], [[:xdigit:]])
  
  # path
  file.jpg   - matches "file.jpg"
             - matches "directory/file.jpg"
             - doesn't match "afile.jpg"
             - doesn't match "directory/afile.jpg"
  /file.jpg  - matches "file.jpg" in the root directory of the remote
             - doesn't match "afile.jpg"
             - doesn't match "directory/file.jpg"
  ```

  related flags are `--exclude, --include, --filter` which is followed by rules, or loading rules from a text file `--exclude-from, --include-from, --filter-from`.

  ```bash
  ### exclude (should not be used with include/filter)
  
  # exclude single file at ~/data/test.jpg
  rclone copy ~/data nuts:hawia/data --exclude=/test.jpg
  
  # exclude by file type
  rclone ls remote: --exclude=*.bak --exclude=*.txt
  
  # exclude ~/data/somedir/
  rclone copy ~/data nuts:hawia/data --exclude=/somedir/ # trailing / is necessary, this implies a dir rule, which is optimized by not listing the directory at all.
  rclone copy ~/data nuts:hawia/data --exclude=/somedir/** # same effect, but this is a file rule. It still list /somedir/, but ignores all files and subfolders under it.
  rclone copy ~/data nuts:hawia/data --exclude=/somedir/* # a file rule, and it will only exclude files under first level of /somedir/. e.g., /somedir/subdir/file will still be included.
  
  # exclude ~/data/pattern*/
  rclone copy ~/data nuts:hawia/data --exclude=/pattern*/
  
  ### include
  # inlcude implies a --exclude ** at last (the flags take effect from end to begin)
  
  rclone copy /vol remote: --include "{A,B}/**"
  # equals to
  rclone copy /vol1/A remote:A
  rclone copy /vol1/B remote:B
  ```

  sample file for `--exclude-from`:

  ```txt
  # exclude-file.txt
  *.bak
  /trial*/
  ```

  then you can use

  ```bash
  rclone copy ~/data nuts:hawia/data --exclude-from exclude-file.txt
  ```

  