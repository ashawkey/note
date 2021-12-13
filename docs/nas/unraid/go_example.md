

### An example `go` settings:

```bash
#!/bin/bash

# start the Management Utility
# /usr/local/sbin/emhttp &
http_proxy="http://127.0.0.1:1087" https_proxy="http://127.0.0.1:1087" /usr/local/sbin/emhttp &

# terminal
echo "export http_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile 
echo "export https_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile

# docker mirrors
mkdir -p /etc/docker
tee /etc/docker/daemon.json <<- "EOF"
{
    "registry-mirrors" : [
        "https://gfqhhvk6.mirror.aliyuncs.com",
        "https://registry.docker-cn.com",
        "http://hub-mirror.c.163.com"
    ]
}
EOF


# jellyfin codec
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

# start frp
# * download the release of frp and copy to flash/config/frp
# * edit the frpc.ini
# cp /boot/config/frp/frpc /usr/bin/frpc
# chmod 777 /usr/bin/frpc
# /usr/bin/frpc -c /boot/config/frp/frpc.ini >> /var/log/frpc.log 2>&1 &

```

