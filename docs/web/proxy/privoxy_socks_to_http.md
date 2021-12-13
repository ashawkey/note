# privoxy

### install

```bash
sudo apt-get install privoxy
```



### config

`/usr/local/etc/privoxy/config`

```bash
# socks5 --> http
forward-socks5 / localhost:1080 . #设置转发到本地的socks5代理客户端端口
listen-address 0.0.0.0:8080 #privoxy暴露的HTTP代理地址，设置 privoxy 监听任意 ip的8080端口
forward 10.*.*.*/ . #内网地址不走代理
forward .abc.com/ . #指定域名不走代理
```



### run

```bash
privoxy config
```



