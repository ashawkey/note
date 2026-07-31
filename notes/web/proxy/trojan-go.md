## trojan-go

Support CDN like Cloudflare to hide your server's IP, or use it even if it's blocked.


### download

https://github.com/p4gefau1t/trojan-go/releases

```bash
mkdir trojan-go && cd trojan-go
wget https://github.com/p4gefau1t/trojan-go/releases/download/v0.10.6/trojan-go-linux-amd64.zip
unzip trojan-go-linux-amd64.zip
```


### Cloudflare settings

Go to `SSL/TLS`:

* set `Overview --> Encryption mode`  to `Full`.

* `Origin server --> Origin Certificates --> Create Certificate`. 

  Copy and paste your `cert.pem & private.key`! Put them on your server, like under `/root/cert/`.


Go to `DNS`:

* add a new A record that parse your domain name to your server IP.
* make sure to use `Proxied`.


### server

A [config](https://p4gefau1t.github.io/trojan-go/basic/config/) example:

```json
{
  "run_type": "server",
  "local_addr": "0.0.0.0", 
  "local_port": 443,
  "remote_addr": "127.0.0.1",
  "remote_port": 80,
  "log_level": 2,
  "log_file": "",
  "password": [
    "your_password" // NEED MODIFICATION
  ],
  "disable_http_check": false,
  "udp_timeout": 60,
  "ssl": {
    "verify": true,
    "verify_hostname": true,
    "cert": "/root/cert/cert.pem", // NEED MODIFICATION
    "key": "/root/cert/private.key", // NEED MODIFICATION
    "key_password": "",
    "cipher": "",
    "curves": "",
    "prefer_server_cipher": false,
    "sni": "your_domain_name", // NEED MODIFICATION
    "alpn": [
      "h2"
    ],
    "session_ticket": true,
    "reuse_session": true,
    "plain_http_response": "",
    "fallback_addr": "",
    "fallback_port": 0,
    "fingerprint": "firefox"
  },
  "tcp": {
    "no_delay": true,
    "keep_alive": true,
    "prefer_ipv4": true
  },
  "mux": {
    "enabled": true,
    "concurrency": 64,
    "idle_timeout": 60
  },
  "websocket": {
    "enabled": true, // true if you want to use it behind CDN
    "path": "/websocket", // should be the same as client
    "host": "your_domain_name" // should be the same to SNI
  },
  "shadowsocks": {
    "enabled": false,
    "method": "AES-128-GCM",
    "password": ""
  }
}
```

Enable as a system service:

```bash
# edit it to make sure the path to trojan-go and config exists.
# change User to root if permission denied.
vim trojan-go/examples/trojan-go.service

# enable systemd service
cp trojan-go/examples/trojan-go.service /etc/systemd/system/
systemctl enable trojan-go
systemctl start trojan-go
systemctl statue trojan-go
```

This will make trojan handle 443 port, but you still need to host NGINX at 80 to accept non-trojan requests with a simple config like:

```nginx
server {
    listen 80 http2; # must enable http/2
    server_name _; # match all

    location / {
        proxy_pass https://bilibili.com;
    }
}
```


To reuse 443 via NGINX streaming, first change trojan config:

```json
{
  "local_addr": "127.0.0.1", 
  "local_port": 10241,
  "remote_addr": "127.0.0.1",
  "remote_port": 10242,
}
```

then host NGINX yourself with config like:

```nginx
user root;
worker_processes auto;
pid /run/nginx.pid;
# include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 768;
    # multi_accept on;
}

### reuse 443 port
stream {
    map $ssl_preread_server_name $backend_name {
        kiui.moe web;
        www.kiui.moe web;
        trojan.kiui.moe trojan;
        default web;
    }

    upstream web {
        server 127.0.0.1:10240;
    }

    upstream trojan {
        server 127.0.0.1:10241; ### trojan local_port
    }

    server {
        listen 443 reuseport;
        listen [::]:443 reuseport;
        proxy_pass $backend_name;
        ssl_preread on;
    }
}

http {
    ### for websocket
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
    gzip on;
	
    ### trojan http --> https
    server {
        listen 80;
        listen [::]:80;
        server_name trojan.kiui.moe;
        return 301 https://trojan.kiui.moe$request_uri;
    }
    

    ### trojan remote_port
    server {
        listen 10242 http2; # must enable http/2
        server_name _; # match all

        location / {
            #proxy_pass https://bilibili.com;
            proxy_pass http://kiui.moe; # redirect to web https
        }
    }
    
    ### web http & https
    server {
        
        listen 80;
        listen [::]:80;
        listen 10240 ssl; # enable ssl
        listen [::]:10240 ssl;

        server_name kiui.moe;

        ssl_certificate /root/cert/cert.pem; # path to your ssl cert
        ssl_certificate_key /root/cert/private.key;

        ssl_session_timeout 5m;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
        ssl_prefer_server_ciphers on;

        location / {
            root /root/home;
        }
    }
    
}
```

restart NGINX:

```bash
systemctl restart nginx
systemctl status nginx
```


### Client

#### trojan-go client

a config:

```json
{
  "run_type": "client",
  "local_addr": "0.0.0.0", 
  "local_port": 1080,
  "remote_addr": "your_domain_name", // NEED MODIFICATION
  "remote_port": 443,
  "log_level": 2,
  "log_file": "",
  "password": [
    "your_password" // NEED MODIFICATION
  ],
  "disable_http_check": false,
  "udp_timeout": 60,
  "ssl": {
    "verify": true,
    "verify_hostname": true,
    "cert": "", // NEED MODIFICATION
    "key": "", // NEED MODIFICATION
    "key_password": "",
    "cipher": "",
    "curves": "",
    "prefer_server_cipher": false,
    "sni": "your_domain_name", // NEED MODIFICATION
    "alpn": [
      "h2"
    ],
    "session_ticket": true,
    "reuse_session": true,
    "plain_http_response": "",
    "fallback_addr": "",
    "fallback_port": 0,
    "fingerprint": "firefox"
  },
  "tcp": {
    "no_delay": true,
    "keep_alive": true,
    "prefer_ipv4": true
  },
  "mux": {
    "enabled": true,
    "concurrency": 64,
    "idle_timeout": 60
  },
  "websocket": {
    "enabled": true, // true if you want to use it behind CDN
    "path": "/websocket", // should be the same as server
    "host": "your_domain_name" // should be the same to SNI
  },
  "shadowsocks": {
    "enabled": false,
    "method": "AES-128-GCM",
    "password": ""
  }
}
```

also start it via `systemd`:

```bash
# edit it to make sure the path to trojan-go and config exists.
# change User to root if permission denied.
vim trojan-go/examples/trojan-go.service

# enable systemd service
cp trojan-go/examples/trojan-go.service /etc/systemd/system/
systemctl enable trojan-go
systemctl start trojan-go
systemctl statue trojan-go
```


#### Clash client (recommended)

get clash:

```bash
wget https://github.com/Dreamacro/clash/releases/download/v1.16.0/clash-linux-amd64-v1.16.0.gz
gz -d clash-linux-amd64-v1.16.0.gz
mv clash-linux-amd64-v1.16.0 clash
chmod 777 clash

# manually get Country.mmdb if automatical download failed
wget https://github.com/Dreamacro/maxmind-geoip/releases/download/20230512/Country.mmdb
```

config like:

```yaml
proxies:
  - name: your_trojan_proxy
    type: trojan
    server: trojan.kiui.moe
    port: 443
    password: your_password
    network: ws
    sni: trojan.kiui.moe
    # udp: true
    # skip-cert-verify: true
    ws-opts:
      path: /websocket
      headers:
        Host: trojan.kiui.moe
```

make it a service under `/etc/systemd/system/clash.service`: (MODIFY the paths!)

```
[Unit]
Description=Clash daemon, A rule-based proxy in Go.
After=network-online.target

[Service]
Type=simple
Restart=always
ExecStart=/path/to/clash -d /path/to/workspace -f /path/to/config

[Install]
WantedBy=multi-user.target
```

enable it:

```bash
systemctl enable clash
systemctl start clash
```

finally, set up system proxy to manual and enter your http/socks5 port.


