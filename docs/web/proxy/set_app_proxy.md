# use proxy



### find the right port !!!

* Windows

​		trojan-qt5 local client's socks port is in "入站设置", not "出站设置".

​		the default is `socks5://127.0.0.1:51837 `

* Ubuntu

  trojan client's default is `socks5://127.0.0.1:1080`



### git

```bash
# set (do not need to set https.proxy!)
git config --global http.proxy socks5://127.0.0.1:51837 

# unset
git config --global --unset http.proxy

# show 
git config --global http.proxy 
```



### bash

```bash
# set
export http_proxy=socks5://127.0.0.1:8080
export https_proxy=socks5://127.0.0.1:8080

# unset
export http_proxy=
export https_proxy=

# show
echo $http_proxy
echo $https_proxy
```



### pip

```bash
# by default, it will use http_proxy and https_proxy

# explitly set
pip install -U --proxy=socks5://127.0.0.1:51837 ddddsr

# note: to use socks proxy, you have to install pysocks first [
pip install pysocks
```





### powershell

(not tested)

```powershell
# set
netsh winhttp set proxy "192.168.0.14:3128"

# unset 

# show
netsh winhttp show proxy
```



### docker

* `docker build`

  * Use `ENV` at Dockerfile (not recommended, since it makes the Dockerfile not portable)

    First check docker's ip via `ip -4 addr`, which is default to `127.17.0.1` in Ubuntu.

    Then add in `Dockerfile`:

    ```dockerfile
    ENV http_proxy "socks5h://127.17.0.1:1080"                                                           ENV https_proxy "socks5h://127.17.0.1:1080"
    ```

    and call with `docker build --network=host -t tag ...`

* `docker run`
