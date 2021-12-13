# IP address

### IPv4

Special IPs:

* `0.0.0.0`

  Non-routable meta address. 

  Used to designate Invalid, unknown targets.

  Also used as host to listen up to all public IPs. (`flask run -h 0.0.0.0`)

* `127.0.0.1`

  one of loopback address (starting from `127`) used to establish connection with the same machine.

* `localhost`

  Usually point to `127.0.0.1`

  Defined in `/etc/hosts`:

  ```
  127.0.0.1 localhost
  ::1 localhost // ipv6
  ```





### IPv6

128bit





### NAT (Network Address Translation)

用于缓解IPv4不足，指的是路由器自动修改IP报文实现LAN/WAN之间转换的过程。

