# uWSGI

### Config

uwsgi.ini

```
[uwsgi]
#配合nginx使用
socket = 127.0.0.1:8000
#项目路径 /Users/xiaoyuan/Desktop/flask_test
chdir = 自己项目路径
#wsgi文件 run就是flask启动文件去掉后缀名 app是run.py里面的Flask对象 
module = run:app
#指定工作进程
processes = 4
#主进程
master = true
#每个工作进程有2个线程
threads = 2
#指的后台启动 日志输出的地方
daemonize = uwsgi.log
#保存主进程的进程号
pidfile = uwsgi.pid
```



### Setup  Nginx

```
server {
	# 监听端口
    listen 80;
    
    # 监听ip 换成服务器公网IP
    server_name 127.0.0.1;
 
	#动态请求
	location / {
	  include uwsgi_params;
	  uwsgi_pass 127.0.0.1:8000;
	}
	
	#静态请求
	location /static {
		alias /Users/xiaoyuan/Desktop/flask_test/static;
	}
}
```





### CLI

```bash
# run
uwsgi --ini uwsgi.ini 

# reload
uwsgi --reload uwsgi.pid

# stop
uwsgi --stop uwsgi.pid

# find pid if deleted uwsgi.pid
ps aux | grep uwsgi
```

