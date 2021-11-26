# ipv6

### Problems

不少校园网采用DHCPv6分配IPv6地址，并且只分配一个`/128`的地址，所以我们需要解决的问题有两个：

* 让路由器下的（包含Unraid服务器在内的）多个设备都获得IPv6。
* 让Unraid服务器内的多个Docker容器都获得IPv6。



### Solutions

实际上，第二个问题可以通过使用容器`Custom: br0`的网络类型来转换成第一个问题，即允许容器直接向路由器请求IPv6地址。但第一个问题似乎只有NAT66能够比较简单地解决。

这里使用OpenWRT路由器为例。

* 通过`opkg`安装nat6插件：`kmod-ipt-nat6 kmod-nf-nat6`。

* 在路由器终端执行：

  ```bash
  echo "net.ipv6.conf.default.accept_ra=2" >> /etc/sysctl.conf
  echo "net.ipv6.conf.all.accept_ra=2" >> /etc/sysctl.conf
  
  uci set network.globals.ula_prefix="$(uci get network.globals.ula_prefix | sed 's/^./d/')"
  uci commit network
  uci set dhcp.lan.ra_default='1'
  uci commit dhcp
  
  touch /etc/hotplug.d/iface/99-ipv6
  
  cat > /etc/hotplug.d/iface/99-ipv6 << EOF
  #!/bin/sh
  [ "\$ACTION" = ifup ] || exit 0
  
  iface_dhcp=wan # modify based on your own internet face name!
  iface_route=wan # modify based on your own internet face name!
  
  [ -z "\$iface_dhcp" -o "\$INTERFACE" = "\$iface_dhcp" ] || exit 0
  
  ip6tables -t nat -I POSTROUTING -s \`uci get network.globals.ula_prefix\` -j MASQUERADE
  gw=\$(ip -6 route show default | grep \$iface_route | sed 's/from [^ ]* //' | head -n1)
  status=\$(ip -6 route add \$gw 2>&1)
  logger -t IPv6 "Done: \$status"
  EOF
  
  /etc/init.d/network restart
  ```

* 等待网络重启完成，

Unraid方面需要在设置中更改网络类型为`IPv4+IPv6`，重启Unraid服务器即可看到分配到的ipv6地址，此时设置中的路由表下应有ipv4和ipv6各三条设置，或者通过终端检查：

```bash
# should have a line like `inet6 xxx scope global xxx`
ip -6 addr
```

对于需要ipv6的Docker应用，需要更改其网络类型为`Custom: br0`，重启后打开对应容器的终端可以用同上的方法检测是否获得了ipv6地址。



### References

* https://kjzjj.com/index.php/2021/09/22/openwrt-ipv6-nat/

* https://blog.191110.xyz/article/000002/.html

* https://post.smzdm.com/p/alpz5z98/

* https://post.smzdm.com/p/awk588rm/
