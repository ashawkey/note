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
  
* check the content without unzipping:

  ```bash
  vim archive.zip # you can even further read the content of a file in it.
  less archive.zip
  
  unzip -l archive.zip # list archive
  unzip -p archive.zip a_certain_file # print content of a file
  ```

  
# 7z

```bash
sudo apt install p7zip-full

7z a out.7z -r folder1 folder2 ...
7z x out.7z
```

