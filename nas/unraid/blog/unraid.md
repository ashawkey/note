# unraid配置



### 安装

很简单，根据[官网](https://unraid.net/)指导即可。



### 配置

* 启动Docker。

* 安装社区应用插件（见下一篇）。

  

### 作为NAS的功能

* 配置磁盘阵列并启动。
* 



### 启动脚本

位于`/boot/config/go`。

需要开机启动的代码都可以放到这个文件内。

例子：

```bash
#!/bin/bash

# start the Management Utility
# /usr/local/sbin/emhttp &
http_proxy="http://127.0.0.1:1087" https_proxy="http://127.0.0.1:1087" /usr/local/sbin/emhttp &

# terminal proxy
echo "export http_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile 
echo "export https_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile

# docker mirrors
mkdir -p /etc/docker
tee /etc/docker/daemon.json <<- "EOF"
{
    "registry-mirrors" : [
        "https://[yourid].mirror.aliyuncs.com",
        "https://registry.docker-cn.com",
        "http://hub-mirror.c.163.com"
    ]
}
EOF

# hardware decoding of video
modprobe i915

# alias 
tee /etc/profile <<- "EOF"
alias l="ls -lrth"
alias la="ls -lrtha"
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias le="less -S"
alias python="python3"
EOF
```

