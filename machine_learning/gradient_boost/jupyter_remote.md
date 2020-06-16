### Jupyter Notebook Remote

``` bash
# server shell
jupyter notebook --no-browser --port=8889 test.ipynb

# local shell
ssh -NfL localhost:8888:localhost:8889 user@server_host

# local browser
localhost:8888
```
