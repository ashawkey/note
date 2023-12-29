# Nginx

### Concepts

* Proxy (Server): An intermediary for requests from clients.
* Reverse Proxy (Server): An intermediary for responses from servers.


### Systemd

`Systemd` is a series of commands, it replaces `initd`  and has `pid=1`.

```bash
systemctl reboot 
systemctl poweroff # halt, suspend, hibernate

# Unit
systemctl list-units

systemctl status
systemctl status bluetooth.service

systemctl is-active application.service
systemctl is-failed application.service
systemctl is-enabled application.service

systemctl start apache.service
systemctl stop apache.service
systemctl restart apache.service
systemctl kill apache.service
systemctl reload apache.service

systemctl enable apache.service # sudo ln -s '/usr/lib/systemd/system/apache.service' '/etc/systemd/system/multi-user.target.wants/apache.service'
systemctl disable apache.service # remove link

systemctl cat apache.service # cat config file
```


### CLI

```bash
systemctl start nginx.service

nginx -s reload
nginx -s stop
nginx -s quit

nginx -t # test, check config files.

```


### Logs

default:

```
/var/log/nginx/access.log
/var/log/nginx/error.log
```


### Configs

Location: `/etc/nginx/nginx.conf`

* Static Website

  ```bash
  # basic
  http {
      server {
      	listen 80;
          location / {
              root /data; # root directory
              index index.html; # http://mysite.com/index.html --> /data/index.html
          }
      }
  }
  
  # root vs alias
  http {
      server {
      	listen 80;
          location /static/ {
          	# location is appended to root
              root /data; # http://mysite.com/static/ --> /data/static/
          }
      }
      server {
      	listen 80;
          location /static/ {
          	# location is dropped, and instead use the alias (the trailing `/` is nececssary here!)
              alias /data/; # http://mysite.com/static --> /data/ 
          }
      }
  }
  
  
  # auto serve static files
  http {
      server {
      	listen 80;
      	# http://mysite.com/files --> /data/files
          location /files {
              root /data; # serve /data/files
  			autoindex on; # automatically show the directory tree (if off, will show 403 Forbidden, but can still access the file using fullname.)
          }
      }
  }
  
  # more detailed example of the trailing `/`
  # access: http://localhost/app/path/file
  # server: /root/app/path/file
  location /app/ {
  	root /root/; # --> /root/app/path/file
  } 
  location /app/ {
  	root /root; # --> /root/app/path/file
  } 
  location /app {
  	root /root/; # --> /root/app/path/file
  } 
  location /app {
  	root /root; # --> /root/app/path/file
  } 
  location /app {
  	alias /root/; # --> /root/path/file
  } 
  location /app {
  	alias /root; # --> (wrong) TODO
  } 
  ```
  
* Dynamic Website (need backend)

  ```bash
  http {
      server {
      	listen 127.0.0.1:8080;
      	
      	# / will redirect to uwsgi port
  		location / {
  		  	  include uwsgi_params;
  			  uwsgi_pass 127.0.0.1:8000;
  		}
  		
      }
  }
  ```
  
* redirect

  ```nginx
  http {
      server {
      	...
          
          # make a shortname for your github repo:
          # kiui.moe/g<else> --> github.com/ashawkey<else>
          location /g {
              rewrite ^/g(.*)$ https://github.com/ashawkey$1 redirect;
          }
      }
  }
  ```

  

* Reverse Proxy

  Proxy to local port:

  ```nginx
  http {
      server {
          # access at domain.com/app/<params> 
          location /app {
  			proxy_pass http://127.0.0.1:5000/; # the trailing `/` means relative --> --> localhost:5000/<params>
              # proxy_pass http://127.0.0.1:5000; # w/o the `/`, it means --> localhost:5000/app/<params>
              
              # to correctly pass everything:
              proxy_set_header  X-Real-IP $remote_addr;
              proxy_set_header  X-Forwarded-For $remote_addr;
              proxy_set_header  X-Forwarded-Proto $scheme;
              proxy_set_header  X-Forwarded-Host $http_host;
              proxy_set_header  Host $host;            
              
          }
      }
  }
  
  # more detailed example of the trailing `/`
  # access http://localhost/app/api/abc
  location /app/ {
      proxy_pass http://localhost:1234/; # --> http://localhost:1234/api/abc (relative)
  }
  location /app {
      proxy_pass http://localhost:1234/; # --> http://localhost:1234//api/abc (relative, better not use)
  }
  location /app/ {
      proxy_pass http://localhost:1234; # --> http://localhost:1234/app/api/abc (absolute)
  }
  location /app {
      proxy_pass http://localhost:1234; # --> http://localhost:1234/app/api/abc (absolute)
  }
  ```

* HTTPS

  ```nginx
  http {
      server {
      	listen 443 ssl; # https usually use 443
      	
      	server_name www.hawia.xyz;
  		
  		# 购买的证书位置
      	ssl_certificate cert.pem;
      	ssl_certificate_key cert.key;
      	
  		ssl_session_timeout 5m;
  		ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
      	ssl_protocols TLSv1 TLSv1.1 TLSv1.2;   #使用该协议进行配置。
  		ssl_prefer_server_ciphers on;  
      	
          location / {
  			proxy_pass http://www.example.com/link/;
          }
      }
  }
  
  # http auto redirect to https
  server {
      listen 80;
      
      server_name localhost;   #将localhost修改为您证书绑定的域名，例如：www.example.com。
      rewrite ^(.*)$ https://$host$1 permanent;   #将所有http请求通过rewrite重定向到https。
      
      location / {
   		index index.html index.htm;
  	}
  }
  ```

* Load Balance

  ```nginx
  http {
  	upstream load_balance_server {
  		server 192.168.1.11:80   weight=5;
  		server 192.168.1.12:80   weight=1;
  		server 192.168.1.13:80   weight=6;
  	}    
      server {
      	...
      }
  }
  ```

