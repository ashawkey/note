# unraid配置



### 安装

很简单，根据[官网指导](https://unraid.net/zh/%E4%B8%8B%E8%BD%BD)即可。



### 配置

同样参考[官方教程](https://wiki.unraid.net/%E5%85%A5%E9%97%A8%E6%8C%87%E5%8D%97_-_Chinese_Getting_Started_Guide)即可。视频教程推荐[这里](https://forums.unraid.net/topic/113327-%E6%9C%80%E8%AF%A6%E5%B0%BD%E7%9A%84unraid%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B/)。

一些基本的设置：

* 日期与时间。
* 启动磁盘阵列，创建共享文件夹。
* 启动Docker。

配置完成后就可以通过NFS/SMB等服务在内网其他机器上进行访问共享文件夹了！

但为了获得更完整的体验，还需要安装各种Docker服务，详见[这里](./docker_app.md)。



### 启动脚本

位于`/boot/config/go`。

需要开机启动的代码都可以放到这个文件内。

仅供参考的例子：

```bash
#!/bin/bash

# start the Management Utility
# /usr/local/sbin/emhttp &
http_proxy="http://127.0.0.1:1087" https_proxy="http://127.0.0.1:1087" /usr/local/sbin/emhttp &

# terminal proxy
echo "export http_proxy=\"http://127.0.0.1:1087\"" >> /etc/profile
echo "export https_proxy=\"http://127.0.0.1:1087\"" >> /etc/profile

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
tee -a /etc/profile <<- "EOF"
alias l="ls -lrth"
alias la="ls -lrtha"
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias le="less -S"
alias python="python3"
EOF
```

