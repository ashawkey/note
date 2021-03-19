# zip

* zip with wild card.

  ```bash
  zip -r out.zip ./*/*/*.jpg
  ```

* exclude sub-directory

  ```bash
  zip -r out.zip <dir> -x "dir/subdir1"  "dir/subdir1/*"
  # should use "", or escape asterisk by \*
  
  # example
zip drn.zip -r drn-master/ -x "drn-master/pretrained/*"
  ```
  
  