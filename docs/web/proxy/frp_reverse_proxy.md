# reverse proxy with `frp`

### install

download from [releases](https://github.com/fatedier/frp/releases).

* `frps`: server
* `frpc`: client


### deploy

* `frps`:

  ```bash
  [common]
  bind_port = 7000 # frps will listen this port for tcp
  vhost_http_port = 8080 # virtual host for forwarded http
  ```

  Run:

  ```bash
  frps -c frps.ini
  ```

  **Don't forget to allow in these ports in firewall** !!!

  ```bash
  ufw allow 7000 # accept frpc communications
  # ufw allow 5000 # do not need since we will use nginx
  ufw allow 2222 # we use this port to forward ssh
  ```

  combine with `nginx` rewrite so we can avoid accessing with port:

  ```nginx
  location /nas {
  	proxy_pass http://127.0.0.1:5000/; # note the backslash! it means relative pass.
      
      # to correctly pass everything: 
      proxy_set_header  X-Real-IP $remote_addr;
      proxy_set_header  X-Forwarded-For $remote_addr;
      proxy_set_header  X-Forwarded-Proto $scheme;
      proxy_set_header  X-Forwarded-Host $http_host;
      proxy_set_header  Host $host;
      proxy_http_version  1.1;
      proxy_set_header    Upgrade $http_upgrade;
      proxy_set_header    Connection $connection_upgrade;
  
  }
  
  # maybe need to add this to http {}
  http {
      ...
      map $http_upgrade $connection_upgrade {
          default upgrade;
          '' close;
      }
      ...
  }
  
  ```

  
* `frpc`:

  ```bash
  [common]
  server_addr = x.x.x.x
  server_port = 7000 # same as frps
  
  # forward ssh (22)
  [ssh]
  type = tcp
  local_ip = 127.0.0.1
  local_port = 22
  remote_port = 6000 # ssh user@x.x.x.x -port 6000 
  
  # forward web service
  [web]
  type = http
  local_port = 80
  custom_domains = yourdomain.com # parse yourdomain.com to x.x.x.x, then access by http://yourdomain.com:8080 
  
  # another web service
  [web2]
  type = http
  local_port = 8080 # another web hosted at localhost:8080
  custom_domains = yourdomain2.com # access by http://yourdomain2.com:8080
  
  # forward as https
  [test_htts2http]
  type = https
  custom_domains = yourdomain3.com # access by https://yourdomain3.com
  
  plugin = https2http
  plugin_local_addr = 127.0.0.1:80
  
  # HTTPS 证书相关的配置
  plugin_crt_path = ./server.crt
  plugin_key_path = ./server.key
  plugin_host_header_rewrite = 127.0.0.1
  plugin_header_X-From-Where = frp
  ```

  Run:

  ```bash
  frpc -c frpc.ini
  ```

  