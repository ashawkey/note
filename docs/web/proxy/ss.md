# Shadowsocks Configuration

Server is CentOS 7 (Digital Oceans)..



### Shadowsocks

edit `/etc/shadowsocks.json`

```json
{
    "server": "0.0.0.0",
    "server_port": 8388,
    "local_port": 1080,
    "password": "yourpasswd",
    "timeout": 600,
    "method": "chacha20-ietf-poly1305"
}
```

```json
{
    "server": "0.0.0.0",
    "local_port": 1080,
    "port_password":{
        "8389": "password1",
        "8390": "password2"
    },
    "timeout": 600,
    "method": "chacha20-ietf-poly1305"
}
```



edit `/etc/systemd/system/shadowsocks.service`

```
[Unit]
Description=Shadowsocks

[Service]
TimeoutStartSec=0
ExecStart=/usr/local/bin/ssserver -c /etc/shadowsocks.json

[Install]
WantedBy=multi-user.target
```

(note the `ssserver`'s location may be different)



run `systemctl` to enable shadowsocks daemon.

```bash
systemctl enable shadowsocks
# systemctl daemon-reload

systemctl start shadowsocks

systemctl status shadowsocks 
systemctl status shadowsocks -l # list all

```



Always remember to check the `logs` if something is wrong!

```bash
/var/log/messages
/var/log/secure
```



### Fail2ban

fail2ban is used to secure the server from brute-force ssh attack.

```bash
last # successed logins
lastb -20 # check last 20 failed logins
```



```bash
#! /bin/bash
set -euxo pipefail

# a simple fail2ban sshd jail
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
ignoreip = 127.0.0.1
bantime = 3600 # 60*60
findtime = 600
maxretry = 5
[sshd]
enabled = true
EOF

systemctl restart fail2ban
systemctl status fail2ban -l
```



> sshd
>
> sshd is the daemon of ssh.



remember to check the `log`.

```bash
/var/log/fail2ban.log
```



check banned IPs & unban IP.

```bash
fail2ban-client status sshd # list banned ips
fail2ban-client set sshd unbanip xxx.xxx.xxx.xxx # unban
```



### set up FirewallD

```bash
systemctl enable firewalld
# shadowsocks 
firewall-cmd --add-port=8838/tcp --permanent
firewall-cmd --add-port=8838/udp --permanent # not necessary
firewall-cmd --reload
```



### change SSH default port 

first add a new port, and test on it (make sure you can log in on that port).

then comment port 22 to disable it.

`/etc/ssh/sshd_config`

```bash
Port 6666
```



```bash
systemctl restart sshd
systemctl status sshd

firewall-cmd --add-port=6666/tcp --permanent
firewall-cmd --add-port=6666/udp --permanent # not necessary
firewall-cmd --reload
```



### Else

* add user

  ```bash
  adduser hawkey
  passwd hawkey
  ```

* misc

  ```bash
  touch file # create new empty file, or change last modified time of the file.
  iptables # basic of firewall
  /etc/shadow # passwd
  ```

* timezone

  ```bash
  tzselect
  # then edit .profile
  ```


