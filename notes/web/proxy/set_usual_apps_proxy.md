# use proxy

Prefer setting globally (bash, powershell) rather than setting each individual program...

### find the right port !!!

clash defaults to `http://127.0.0.1:7890` and `socks5://127.0.0.1:7890`

trojan-qt5 local client's socks port is in "入站设置", not "出站设置".	the default is `socks5://127.0.0.1:51837`.


### powershell

```powershell
# set
$Env:http_proxy="http://127.0.0.1:7890"
$Env:https_proxy="http://127.0.0.1:7890"

# unset
$Env:http_proxy=""
$Env:https_proxy=""

# show
echo $Env:http_proxy
echo $Env:https_proxy
```


### bash

```bash
# set
export http_proxy=socks5://127.0.0.1:1080
export https_proxy=socks5://127.0.0.1:1080

export http_proxy=http://127.0.0.1:1081
export https_proxy=http://127.0.0.1:1081

# unset
export http_proxy=
export https_proxy=

# show
echo $http_proxy
echo $https_proxy
```

### wget

```bash
# use .wgetrc
cat > ~/.wgetrc <<EOF
http_proxy=http://127.0.0.1:1081
https_proxy=http://127.0.0.1:1081
HTTP_PROXY=http://127.0.0.1:1081
HTTPS_PROXY=http://127.0.0.1:1081
use_proxy=on
EOF
```

### jupyter notebook

It should use the system proxy, if not working, add this code block and run it:

```python
import os
os.environ['http_proxy'] = "http://127.0.0.1:1081" 
os.environ['https_proxy'] = "http://127.0.0.1:1081" 
```

### git

```bash
# set (do not need to set https.proxy!)
git config --global http.proxy socks5://127.0.0.1:51837 

# unset
git config --global --unset http.proxy

# show 
git config --global http.proxy 
```


### pip

by default, it will use http_proxy and https_proxy, so you can just set these env variables.

```bash
# explitly set
pip install -U --proxy=socks5://127.0.0.1:51837 ddddsr

# note: to use socks proxy, you have to install pysocks first
pip install pysocks
```


### docker

* `docker build`

  * Use `ENV` at Dockerfile (not recommended, since it makes the Dockerfile not portable)

    add in `Dockerfile`:

    ```dockerfile
    # socks5
    ENV http_proxy "socks5h://localhost:1080"
    ENV https_proxy "socks5h://localhost:1080"
    
    # http (recommended)
    ENV http_proxy "http://localhost:1081"
    ENV https_proxy "http://localhost:1081"
    ```

    and call with `docker build --network=host -t tag ...`

* `docker run`


### Authorization

```bash
# for authorization, use proxy like:
http://<user>:<pwd>@<ip>:<port>

# note that special characters in user/pwd should be replaced:
@          %40
:          %3A
!          %21
#          %23
$          %24
%          %25
&          %26
+          %2B
/          %2F
\          %5C
=          %3D
?          %3F
spacebar   %20
.          %2E
```

