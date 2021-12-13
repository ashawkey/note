### `frpc.ini`

```ini
[common]
server_addr = 149.28.65.26
server_port = 7000

[ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = 2222

[web]
type = http
local_port = 5000
custom_domains = kiui.moe
```

### `run_frpc.sh`

```bash
nohup /root/frp/frpc -c /root/frp/frpc.ini </dev/null >/root/frp/log.txt 2>&1 &
```



### `frps.ini`

```ini
[common]
bind_port = 7000
vhost_http_port = 5000
# for a web frontend monitor:
dashboard_port = 7500
dashboard_user = <user>
dashboard_pwd = <pwd>

```



### `nginx.conf`

```nginx
location /nas { # access at kiui.moe/nas/
    include wbo_proxy_headers.conf;
    proxy_pass http://127.0.0.1:5000/;
}
```

