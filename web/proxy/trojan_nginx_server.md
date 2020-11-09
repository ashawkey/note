# Trojan with Nginx 

### Server

* Get trojan (as root)

  ```bash
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/trojan-gfw/trojan-quickstart/master/trojan-quickstart.sh)"
  ```

  This will install trojan at `/usr/local/bin/trojan`, 

  with the config at `/usr/local/etc/trojan/config.json`,

  and it will generate system unit at `/etc/systemd/system/trojan.service`.

* Config

  ```json
  {
      "run_type": "server",
      "local_addr": "127.0.0.1", # modified
      "local_port": 10241, # modified, same as nginx trojan upstream
      "remote_addr": "127.0.0.1",
      "remote_port": 80,
      "password": [
          "password" # modified
      ],
      "log_level": 1,
      "ssl": {
          "cert": "/path/to/certificate.crt", # modified
          "key": "/path/to/private.key", # modified
          "key_password": "",
          "cipher": "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384",
          "cipher_tls13": "TLS_AES_128_GCM_SHA256:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_256_GCM_SHA384",
          "prefer_server_cipher": true,
          "alpn": [
              "http/1.1"
          ],
          "reuse_session": true,
          "session_ticket": false,
          "session_timeout": 600,
          "plain_http_response": "",
          "curves": "",
          "dhparam": ""
      },
      "tcp": {
          "prefer_ipv4": false,
          "no_delay": true,
          "keep_alive": true,
          "reuse_port": false,
          "fast_open": false,
          "fast_open_qlen": 20
      },
      "mysql": {
          "enabled": false,
          "server_addr": "127.0.0.1",
          "server_port": 3306,
          "database": "trojan",
          "username": "trojan",
          "password": "",
          "key": "",
          "cert": "",
          "ca": ""
      }
  }
  ```

* DNS

  Add an A-record for a prefix URL that trojan uses.

  ```bash
  A-record trojan [server_ip] 1min # trojan.your_url.domain
  ```

* Nginx stream

  ```nginx
  ...
  
  # add this
  stream {
      map $ssl_preread_server_name $backend_name {
          hawia.xyz web;
          trojan.hawia.xyz trojan; # map trojan.hawia.xyz to upstream trojan
          default web;
      }
  
      upstream web {
          server 127.0.0.1:10240; # web should listen to this
      }
  
      upstream trojan {
          server 127.0.0.1:10241; # trojan should listen to this
      }
  	
      # stream server
      server {
          listen 443 reuseport;
          listen [::]:443 reuseport;
          proxy_pass $backend_name;
          ssl_preread on;
      }
  }
  
  http {
  	...
  }
  ```

  change web servers to the new port:

  ```nginx
  server {
  
      listen 10240 ssl default_server; # modified from 443
      listen [::]:10240 ssl default_server; # modified from 443
  
      server_name www.hawia.xyz;
  
      ssl_certificate www.hawia.xyz.pem;
      ssl_certificate_key www.hawia.xyz.key;
  
      ssl_session_timeout 5m;
      ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
      ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
      ssl_prefer_server_ciphers on;
  
      location / {
          root /root/home;
      }
  
      location /blogs {
          root /root;
      }
  
      location /nonsense {
          root /root;
      location /umbra {
          root /root;
      }
  
      location /api/ {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:8000;
      }
  
      location /api/nonsense/ {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:8001;
      }
  
      location /api/umbra/ {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:8002;
      }
  }
  
  # http redirect
  server {
      listen 80;
      server_name www.hawia.xyz;
      rewrite ^(.*)$ https://$host$1 permanent;
  }
  ```

* Start it!

  ```bash
  #systemctl start nginx
  nginx -s reload
  systemctl start trojan
  systemctl status trojan
  ```

  



### Client

* Get trojan

  ```bash
  wget https://github.com/trojan-gfw/trojan/releases/download/v1.15.1/trojan-1.15.1-linux-amd64.tar.xz
  tar -xvf trojan-1.15.1-linux-amd64.tar.xz
  ```

* Config

  ```bash
  vi trojan/config.json
  ```

  ```json
  {
      "run_type": "client", # modified
      "local_addr": "0.0.0.0",
      "local_port": 1080,
      "remote_addr": "trojan.hawia.xyz", # modified
      "remote_port": 443, # may need to modify
      "password": [
          "password" # modified
      ],
      "log_level": 1,
      "ssl": {
          "verify": false, # modified
          "verify_hostname": false, # modified
          "cert": "", # modified
          "key": "", # modified
          "key_password": "",
          "cipher": "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-A
          "cipher_tls13": "TLS_AES_128_GCM_SHA256:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_256_GCM_SHA384",
          "prefer_server_cipher": true,
          "alpn": [
              "http/1.1"
          ],
          "alpn_port_override": {
              "h2": 81
          },
          "reuse_session": true,
          "session_ticket": false,
          "session_timeout": 600,
          "plain_http_response": "",
          "curves": "",
          "dhparam": ""
      },
      "tcp": {
          "prefer_ipv4": false,
          "no_delay": true,
          "keep_alive": true,
          "reuse_port": false,
          "fast_open": false,
          "fast_open_qlen": 20
      },
      "mysql": {
          "enabled": false,
          "server_addr": "127.0.0.1",
          "server_port": 3306,
          "database": "trojan",
          "username": "trojan",
          "password": "",
          "cafile": ""
      }
  }
  
  ```

* Run trojan service!

  ```bash
  trojan -c config.json -l trojan.log
  ```

  Or make a system unit.

  ```bash
  cat > /etc/systemd/system/trojan.service <<-EOF
  [Unit]
  Description=trojan
  After=network.target
  
  [Service]
  Type=simple
  PIDFile=/usr/src/trojan/trojan.pid
  ExecStart=/usr/src/trojan/trojan -c /usr/src/trojan/config.json -l /usr/src/trojan/trojan.log
  ExecReload=/bin/kill -HUP \$MAINPID
  Restart=on-failure
  RestartSec=1s
  
  [Install]
  WantedBy=multi-user.target
  
  EOF
  ```

  ```bash
  systemctl start trojan
  systemctl status trojan
  systemctl enable trojan
  ```

* Proxychains

  ```bash
  apt install proxychains
  ```

  Config:

  ```bash
  vi /etc/proxychains.conf
  ```

  ```
  ...
  # socks4 ...
  socks5 127.0.0.1 1080
  ```

* test

  ```bash
  curl -4 ip.sb
  proxychains curl -4 ip.sb
  ```

  