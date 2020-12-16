# rsync

* Copy with exclusion

  ```bash
  # copy source to destination, excluding source/folder
  rsync -av --progress source/ destination --exclude folder
  ```

  

* Super fast remove (for large and recursive directories)

  ```bash
  mkdir empty_dir
  rsync -a --delete empty_dir/ dir_to_delete/ # much more faster than rm -rf
  rm -rf empty_dir
  ```

  