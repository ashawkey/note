# Community Applications: The Ultra Proxy Guide.


### Problems

Unraid plugins rely on GitHub. Specifically, it needs to access `github.com` and `raw.githubusercontent.com` to fetch data.


### manually Install CA

* download the plugin in your PC:

  ```
  https://raw.githubusercontent.com/Squidly271/community.applications/master/plugins/community.applications.plg
  ```

* name it `CA.plg` and upload to `/boot/config/plugins/`.

* go to Plugin Tab and install from the local file. It should still fail when downloading `community.applications-***.txz`.
* manually download from the URL and upload it to `/boot/config/plugins/community.applications/`. (The path can be found in the source of `plg` by searching `<FILE` tag).
* go to Plugin Tab and install from the local file.


> Ref: https://post.smzdm.com/p/ad2dngvn/


### Use docker to setup proxy

Since the terminal of unraid do not support package manager (and the alternative  `nerdpack` is also a plugin), we would like to use docker to setup `trojan` proxy.

* set up docker mirrors:

  ```bash
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
  ```

  
* prepare the `config.json` for `trojan` and put it to `/boot/trojan/config.json`. The `local_port` must be `1086`!

  ```json
  {
      "run_type": "client",
      "local_addr": "127.0.0.1",
      "local_port": 1086,
      "remote_addr": "[your domain]",
      "remote_port": 443,
      "password": [
          "[yourpasswd]"
      ],
      "log_level": 1,
      "ssl": {
          "verify": false,
          "verify_hostname": false,
          "cert": "",
          "key": "",
          "key_password": "",
          "cipher": "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384",
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
          "key": "",
          "cert": "",
          "ca": ""
      }
  }
  ```

* pull and start the prepared [image](https://hub.docker.com/repository/docker/ashawkey/trojan-privoxy-client) in terminal:

  ```bash
  docker pull ashawkey/trojan-privoxy-client
  
  # the default proxy:
  # socks5://127.0.0.1:1086
  # http://127.0.0.1:1087
  docker run -d --name tpc -v /boot/config/trojan:/etc/trojan -p 1086:1086 -p 1087:1087 ashawkey/trojan-privoxy-client
    
  # for custom config path and port:
  # docker run -d --name tpc -v $config_dir:/etc/trojan -p $socks5_port:1086 -p $http_port:1087 ashawkey/trojan-privoxy-client
  ```

  you could see the docker running smoothly in the Docker Tab. 

* Turn on auto-restart of this container.


### Configure proxy settings for CA

* Apps Tab: use `curl` to fetch.

  CA plugin provides an explicit `proxy.cfg` to set the `curl` proxy (The code is located [here](https://github.com/Squidly271/community.applications/blob/722f7f489dfbc71382e6dc4a524ee013e29cb344/source/community.applications/usr/local/emhttp/plugins/community.applications/include/helpers.php#L63)).

  Create `/flash/config/plugins/community.applications/proxy.cfg` and put:

  ```
  port=1087
  tunnel=1
  proxy=http://127.0.0.1
  ```

  
* Install Plugin: use `wget` to fetch (The code can be found in `/usr/local/sbin/plugin`).

  Add to `/boot/config/go`:

  ```bash
  # emhttp
  http_proxy="http://127.0.0.1:1087" https_proxy="http://127.0.0.1:1087" /usr/local/sbin/emhttp &
  
  # terminal
  echo "export http_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile 
  echo "export https_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile
  ```

  
* Reboot and you are free to load Apps and install Plugins!


