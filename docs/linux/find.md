# find

* find by name matching

  ```bash
  find . -name "*.sh"
  
  # find multiple directories
  find /usr /home /tmp -name "*.sh" 
  
  # find only directories
  find . -type d "__pycache__"
  
  # find only files
  find . -type f "*pattern*"
  ```

  

* list recursively with max-depth, sort output

  ```bash
  # max-depth = 2
  find . -maxdepth 2 | sort
  ```


* Replace string in all files under a folder

  ```bash
  # replace 'oldtext' to 'newtext' in all files under dir
  grep -rl oldtext dir | xargs sed -i 's/oldtext/newtext/g'
  # or
  find dir -type f -exec sed -i 's/oldtext/newtext/g' {} +
  ```

  

* find all files that contain a string

  ```bash
  grep -rnw -e 'pattern' . # in current dir
  
  # exclude
  grep -rnw -e 'pattern' --exclude '*.py' .
  ```

  

* delete recursively a certain patterned file

  ```sh
  # first find them and print them
  find . -name "pattern" | xargs
  
  # delete them
  find . -name "pattern" | xargs rm -rf
  
  # delete with exclusion
  find . -name "pattern" | grep -v "dont.delete.me" | xargs rm -rf
  
  ```

  