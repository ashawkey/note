# rsync

* general

  ```bash
  -a: archive mode, equals, -rlptgoD, in a word, it is recursive and preserve almost everything from source to target.
  -v: verbose
  -P: equals --progress --partial, show progress, and allow resume.
  ```

  

  

* Copy with exclusion

  ```bash
  # copy source to destination, excluding source/folder
  rsync -avP source/ destination --exclude folder
  
  # exclude multiple
  rsync -avP source/ destination --exclude folder --exclude folder2
  ```




* replace `scp`

  ```bash
  # d
  rsync -avP -e ssh local_file user@root:remote_file
  # custom ssh port
  rsync -avP -e "ssh -p 23" local_file user@root:remote_file
  ```

  



* Super fast remove (for large and recursive directories)

  ```bash
  mkdir empty_dir
  rsync -a --delete empty_dir/ dir_to_delete/ # much more faster than rm -rf
  rm -rf empty_dir
  ```

  