# Trojan with Nginx 

### Server

* Get trojan 

  ```bash
  wget https://github.com/trojan-gfw/trojan/releases/download/v1.16.0/trojan-1.16.0-linux-amd64.tar.xz
  tar xf trojan-1.16.0-linux-amd64.tar.xz
  rm trojan-1.16.0-linux-amd64.tar.xz
  cd trojan
  ```

* Edit `config.json`, set port, password, and cert/key.

  ```bash
  vi config.json
  ```

* register as a service for auto-start when reboot

  ```bash
  sudo vi /etc/systemd/system/trojan.service
  ```

  ```bash
  [Unit]
  Description=trojan
  After=network.target network-online.target nss-lookup.target mysql.service mariadb.service mysqld.service
  
  [Service]
  Type=simple
  StandardError=journal
  ExecStart=/home/ubuntu/trojan/trojan /home/ubuntu/trojan/config.json # CHANGE to your path!
  ExecReload=/bin/kill -HUP \$MAINPID
  LimitNOFILE=51200
  Restart=on-failure
  RestartSec=1s
  
  [Install]
  WantedBy=multi-user.target
  ```

  ```bash
  sudo systemctl start trojan
  sudo systemctl enable trojan
  sudo systemctl status trojan
  sudo netstat -antp # you can find 80 and 443 are in LISTEN status
  ```

* config DNS

  Using Cloudflare, create a new A-record: `your.domain.name` to `your server IP`, and set the mode to `DNS only`.

* start Nginx

  ```bash
  sudo apt install nginx
  sudo vi /etc/nginx/sites-enabled/default
  ```

  ```nginx
  # add a server for fake 80
  server {
      listen 80 default_server;
      server_name <your.domain.name>;
  
      # the camouflage website to redirect
      location / {
          proxy_pass https://bilibili.com;
      }
  
  }
  
  # change the original server to redirect http --> https
  server {
      listen 80; # REMOVE default_server!
      listen [::]:80;
  
      server_name _; # catch-all, any other server_name will use this.
  
      location / {
          return 301 https://$host$request_uri;
      }
  }
  ```

  ```bash
  sudo systemctl restart nginx
  sudo systemctl status nginx
  ```

* And now everything is set:

  * access `your.domain.name`, you'll jump to the camouflage site.

  * using a trojan/clash/... client to connect! 

    
#### Nginx reuse 443

this happens if we also use Nginx to host other websites.

Nginx listens to 443 and stream the requests according to server names.

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
# trojan http --> https
server {
    listen 80;
    listen [::]:80;
    server_name x.kiui.moe;
    return 301 https://x.kiui.moe$request_uri;
}

# trojan remote_port (fake web)
server {
    listen 10242;
    server_name x.kiui.moe;

    location / {
        proxy_pass https://bilibili.com;
    }
}

# web http --> https
server {
    listen 80;
    listen [::]:80;
    server_name kiui.moe;
    return 301 https://kiui.moe$request_uri;
}

# web https
server {

    listen 10240 ssl; # modified from 443
    listen [::]:10240 ssl; # modified from 443

    server_name hawia.xyz;

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
```

* Start it!

  ```bash
  #systemctl start nginx
  nginx -s reload
  systemctl start trojan
  systemctl status trojan
  ```

* If use Cloudflare CDN:

  change the trojan record to `DNS Only` status. (do not proxy)

  ![image-20210305154420993](trojan_nginx_server.assets/image-20210305154420993.png)


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
          "password" # modified!
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

  Or make a system unit. (**MODIFY the trojan path!**)

  ```bash
  cat > /etc/systemd/system/trojan.service <<-EOF
  [Unit]
  Description=trojan
  After=network.target
  
  [Service]
  Type=simple
  PIDFile=/home/kiui/trojan/trojan.pid
  ExecStart=/home/kiui/trojan/trojan -c /home/kiui/trojan/config.json -l /home/kiui/trojan/trojan.log
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

* Privoxy for http proxy

  since trojan only supports socks5 proxy, you should use `privoxy` to forward it to `http` proxy:

  ```bash
  sudo apt install privoxy
  ```

  edit config `/etc/privoxy/config`, add the following lines:

  ```bash
  listen-address 0.0.0.0:1081 # http proxy port
  toggle  1
  enable-remote-toggle 1
  enable-remote-http-toggle 1
  enable-edit-actions 0
  enforce-blocks 0
  buffer-limit 4096
  forwarded-connect-retries  0
  accept-intercepted-requests 0
  allow-cgi-request-crunching 0
  split-large-forms 0
  keep-alive-timeout 5
  socket-timeout 60
  
  forward-socks5 / 0.0.0.0:1080 . # trojan's socks5 proxy port
  ```

  restart the service:

  ```bash
  sudo systemctl restart privoxy
  ```

  Now you can check the proxy via:

  ```bash
  sudo netstat -antp
  ```

  