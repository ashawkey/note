# ip

### basics

```bash
# show all network interfaces & IP address
ip addr [show]

ip link # only show l
ip -4 addr # only ipv4
ip -6 addr # only ipv6
```

Output:

```bash
### example output of `ip addr`

# loopback (127.0.0.1, ::1) 用于本主机内部通信。
# LOWER_UP表示有网线接入网络接口，UP表示网络接口正在运行
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
	# 网卡 MAC地址 broadcast地址
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    # ipv4地址（CIDR），scope host表示仅在本地通信
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    # ipv6地址
    inet6 ::1/128 scope host
       valid_lft forever preferred_lft forever
# 第一个网络接口
2: enp129s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether ac:1f:6b:27:b1:a8 brd ff:ff:ff:ff:ff:ff
    # scope global表示开放外网通信
    inet 222.29.2.45/24 brd 222.29.2.255 scope global enp129s0f0
       valid_lft forever preferred_lft forever
    # scope dynamic表示ip动态获取
    inet6 2001:da8:201:1302:ae1f:6bff:fe27:b1a8/64 scope global mngtmpaddr dynamic
       valid_lft 2591921sec preferred_lft 604721sec
    # scope link表示仅在内网通信
    inet6 fe80::ae1f:6bff:fe27:b1a8/64 scope link
       valid_lft forever preferred_lft forever
# 第二个网络接口（未插入网线，没有LOWER_UP）
3: enp129s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether ac:1f:6b:27:b1:a9 brd ff:ff:ff:ff:ff:ff
# docker的网络接口
4: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default
    link/ether 02:42:db:df:da:5f brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
    inet6 fe80::42:dbff:fedf:da5f/64 scope link
       valid_lft forever preferred_lft forever

```

