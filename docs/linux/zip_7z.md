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
  
  



# 7z

```bash
sudo apt install p7zip-full

7z a out.7z -r folder1 folder2 ...
7z x out.7z
```

