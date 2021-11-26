# Community Applications: The Ultimate Proxy Guide



### Problems

Unraid Community Applications (CA) Plugin relies on GitHub to download data.

Specifically, it needs to access `github.com` and `raw.githubusercontent.com` to fetch data.

However, from this [wiki](https://zh.wikipedia.org/wiki/%E5%AF%B9GitHub%E7%9A%84%E5%AE%A1%E6%9F%A5%E5%92%8C%E5%B0%81%E9%94%81), simply modifying `Hosts` is not enough now.

This article provides a way to access CA via proxy, using [trojan-gfw](https://trojan-gfw.github.io/trojan/) as an example.



### Manually Install CA

* download the plugin in your PC (which can access github):

  ```
  https://raw.githubusercontent.com/Squidly271/community.applications/master/plugins/community.applications.plg
  ```

* name it `community.applications.plg` and upload it to `/boot/config/plugins/`.

* go to the Plugin Tab and choose the uploaded plugin file and install locally.

  It should still fail when downloading `community.applications-***.txz`.

* manually download that file from the URL and upload it to `/boot/config/plugins/community.applications/`. 

  (The path can be found in the source of `plg` by searching `<FILE` tag).

* again, go to Plugin Tab and install from local file.

  It should success now.



> Ref: https://post.smzdm.com/p/ad2dngvn/



### Setup proxy via docker

Since the terminal of Unraid do not support package manager (and the alternative  `nerdpack` is also a plugin), we would like to use docker to setup `trojan` proxy.

We use trojan-gfw as an example, if you use other tools, you should build your own docker image.

* set up docker mirrors for faster downloading:

  (You may optionally use your own mirror provided by aliyun.)

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

* prepare the `config.json` for `trojan` and put it at `/boot/trojan/config.json`. The `local_port` must be `1086`:

  An example, do not use without modification:

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
  ```

  You should see the docker running smoothly in the Docker Tab now! Optionally you can turn on `always restart`.



### Configure proxy settings

There are two things we need to care: the Apps Tab Information and the Plugin Install process.

* Apps Tab Information

  From the [code](https://github.com/Squidly271/community.applications/blob/722f7f489dfbc71382e6dc4a524ee013e29cb344/source/community.applications/usr/local/emhttp/plugins/community.applications/include/helpers.php#L63), it uses `curl` to fetch information. 

  Besides, it provides a `proxy.cfg` file to explicitly set  up the proxy for `curl`. 

  We will just use this! Create `/boot/config/plugins/community.applications/proxy.cfg` and write:

  ```
  port=1087
  tunnel=1
  proxy=http://127.0.0.1
  ```

  `curl` supports both http and socks5 proxy, we use the http proxy here.

  

* Install Plugin: 

  From the code (located at `/usr/local/sbin/plugin` from the terminal), it uses `wget` to download plugins.

  `wget` only supports http proxy, so we need to add the following lines to `/boot/config/go`:

  ```bash
  # emhttp
  http_proxy="http://127.0.0.1:1087" https_proxy="http://127.0.0.1:1087" /usr/local/sbin/emhttp &
  
  # terminal
  echo "export http_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile 
  echo "export https_proxy=\"http://127.0.0.1:1087\"" >> /root/.bash_profile
  ```

  

Now reboot and you are free to load Apps and install Plugins!

If not, you should look at the logs of the docker container to debug which step goes wrong.



