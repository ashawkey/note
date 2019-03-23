# 域名解析

### 域

* `com` 顶级域名

* `hawspiral` 二级域名

* 在此之下可以任意添加三级域名，形如`*.hawspiral.com`

  * `www`：著名的www子域名


### GFW

* Local DNS缓存污染

  指向固定的无效IP

* IP封锁

  针对拥有大量连续地址段的ISP(Internet Service Provider)进行封锁

* TCP重置

  审查TCP packet内容，比链接双方更快的发送RESET信息

* ...



### 域名解析服务

DNS服务商提供域名解析服务。

每条记录包括**记录类型，主机记录**，解析线路ISP，**记录值**，TTL，MX优先级。



* **A记录**：IP指向（Alias），即把域名指向地址。

  * 把整个二级域名指向服务器IP

    同一个域名可以指向多个IP地址，优先使用上面的。

    * 主机记录：@（留空）
    * 记录值：IPv4地址

  * 设置`www`子域名
    * 主机记录: `www`

    * 记录值:`example.com`

* **CNAME记录**：别名指向（Canonical name）

  A记录优先于CNAME记录，所以无法强行把已有A记录的域名指向另一个域名。

  * 把`www`子域名指向`example.github.io`

    * 主机记录: `www`

    * 记录值: `example.github.io`

* MX记录：邮件交换(Mail Exchanger)

* NS记录：解析服务器

* TXT记录：对域名的说明

  常用于SPF反垃圾邮件。m

