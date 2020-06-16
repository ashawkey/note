# nginx static website

### Start nginx service

安装nginx之后，nginx会自动在后台运行。更改配置文件之后只需要使用reload热重启即可。

```bash
# root
systemctl status nginx -l # centos 7 
nginx -s reload # hot reload
```



### Example: host a static website



### Configuration files in detail

```bash
# nginx.conf
# the main configuration file. 
```





### Trouble shooting

##### `bind() to 80 failed (98: Address already in use)`

* 配置文件语法

* 端口占用

  ```bash
  netstat -tlnp | grep 80
  # check port 80 usage
  # -t: tcp
  # -l:
  # -n:  
  # -p: show program
  
  fuser -k 80/tcp
  # kill all processes using 80/tcp
  ```




