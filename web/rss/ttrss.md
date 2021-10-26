# TTRSS



### install

```bash
# install docker
# https://docs.docker.com/engine/install/

# install docker-compose
# https://docs.docker.com/compose/install/

# install ttrss
# http://ttrss.henry.wang/zh/#%E5%85%B3%E4%BA%8E
mkdir ttrss && cd ttrss
wget https://github.com/HenryQW/Awesome-TTRSS/blob/main/docker-compose.yml

# modify following the yml
# hint1: SELF_URL_PATH: only the domain name! and use https.

# run at foreground
docker-compose up
```



### Nginx

As sub-domain mode: 

* recommend, because ttrss will also create other sub-directories... so you don't want to pollute your sub-dir, right?
* however, you need to issue a new ssl certicifate for the sub-domain. :-(

```nginx
# upstream redirected to share 443
# access at ttrss.kiui.moe
server {
    listen 10243 ssl;
    listen [::]:10243 ssl;
    
    gzip on;
    server_name  ttrss.kiui.moe;

    ssl_certificate ttrss.fullchain.pem;
    ssl_certificate_key ttrss.privkey.pem;

    location / {
        proxy_redirect off;
        proxy_pass http://127.0.0.1:181;

        proxy_set_header  Host                $http_host;
        proxy_set_header  X-Real-IP           $remote_addr;
        proxy_set_header  X-Forwarded-Ssl     on;
        proxy_set_header  X-Forwarded-For     $proxy_add_x_forwarded_for;
        proxy_set_header  X-Forwarded-Proto   $scheme;
        proxy_set_header  X-Frame-Options     SAMEORIGIN;

        client_max_body_size        100m;
        client_body_buffer_size     128k;

        proxy_buffer_size           4k;
        proxy_buffers               4 32k;
        proxy_busy_buffers_size     64k;
        proxy_temp_file_write_size  64k;
    }
}
```



As sub-directory mode: [DO NOT RECOMMEND!]

* If use this mode, all the APIs in the web frontend needs to be modified. e.g., the fever API should be `https://kiui.moe/ttrss/plugins/fever`, but the web frontend shows `https://kiui.moe/plugins/fever/`. This will also lead to several bugs.

```nginx
server {
    # ...
    
    # access at kiui.moe/ttrss/
    # SELF_URL_PATH = https://kiui.moe/
    location /ttrss/ {
        rewrite ^/ttrss/(.*)$ /$1 break;
        proxy_redirect https://$http_host https://$http_host/ttrss;
        proxy_pass http://127.0.0.1:181; # local service at 181

        proxy_set_header  Host                $http_host;
        proxy_set_header  X-Real-IP           $remote_addr;
        proxy_set_header  X-Forwarded-Ssl     on;
        proxy_set_header  X-Forwarded-For     $proxy_add_x_forwarded_for;
        proxy_set_header  X-Forwarded-Proto   $scheme;
        proxy_set_header  X-Frame-Options     SAMEORIGIN;

        client_max_body_size        100m;
        client_body_buffer_size     128k;

        proxy_buffer_size           4k;
        proxy_buffers               4 32k;
        proxy_busy_buffers_size     64k;
        proxy_temp_file_write_size  64k;
    }
}
```





### access web frontend

At `kiui.moe/ttrss/` or `ttrss.kiui.moe`. 

Default account is `admin`. Remember to change password at once.



### fever api for 3rd applications

* activate in prefs.
* set a new password.

