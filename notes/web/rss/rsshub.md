# RSSHub service

We use the [**manual**](https://docs.rsshub.app/install/#shou-dong-bu-shu) install to gain full control.

```bash
# install & update nodejs & npm
sudo apt install nodejs npm
sudo npm install -g n
sudo n stable
sudo npm install -g npm

# install RSSHub
git clone https://github.com/DIYgod/RSSHub.git
cd RSSHub
npm ci --production

# start 
npm start
```


### configs

create `.env` file to write configs.

```bash
# port, default is 1200
PORT=1888

# access control

```


### [Nginx forward](https://gist.github.com/soheilhy/8b94347ff8336d971ad0)

```nginx
server {
    # ...
    
    # serve at kiui.moe/rss
    location /rss {
        # kiui.moe/rss/... --> localhost:1888/...
        rewrite ^/rss/(.*)$ /$1 break;
        proxy_pass http://127.0.0.1:1888;
    }
}
```


