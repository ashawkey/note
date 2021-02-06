# zip

* zip with wild card.

  ```bash
  zip -r out.zip ./*/*/*.jpg
  ```

* exclude sub-directory

  ```bash
  zip -r out.zip <dir> -x <dir-to-exclude-1> <dir-to-exclude-2>
  # should use "dir-to-exclude" 
  # or escape asterisk by \*
  ```

  