# network utils

### ifconfig

Gathering network related data.

```bash
# example output of ifconfig
eno1      Link encap:Ethernet  HWaddr 0c:c4:7a:e3:5b:fe
          inet addr:222.29.2.44  Bcast:222.29.2.255  Mask:255.255.255.0
          inet6 addr: 2001:da8:201:1302:ec4:7aff:fee3:5bfe/64 Scope:Global
          inet6 addr: fe80::ec4:7aff:fee3:5bfe/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1000529332 errors:0 dropped:310819822 overruns:3907 frame:0
          TX packets:1016086867 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:291468156454 (291.4 GB)  TX bytes:1366810427542 (1.3 TB)
          Memory:fb120000-fb13ffff

eno2      Link encap:Ethernet  HWaddr 0c:c4:7a:e3:5b:ff
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
          Memory:fb100000-fb11ffff

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:157678206 errors:0 dropped:0 overruns:0 frame:0
          TX packets:157678206 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:2013281906725 (2.0 TB)  TX bytes:2013281906725 (2.0 TB)

```

Explain:

* `Link encap`: link encapsulation. e.g., Ethernet, Local Loopback.

  almost all encapsulations are Ethernet.

  > I don’t know what the networking protocol of the future is, but it will be called ‘Ethernet.’

* `HWaddr`:  hardware address (MAC address)

  unique to each ethernet card.

* `inet addr`: IP address

* `Bast`: broadcast address

* `Mask`: network mask

* `UP`: kernel modules related to ethernet interface.

  * `BROADCAST`: support broadcast
  * `NOTRAILERS`: trailer encapsulation is disabled. Linux usually ignore this.
  * `RUNNING`: ready to accept data.
  * `MULTICAST`: support multicast.

* `MTU`: Maximum Transmission Unit. (size of each packet received by the ethernet card)

* `Metric`: priority of this device.

* `RX Packets, TX Packets`: total number of received, transmitted packets.

* `collisions`: collisions for congestion, should ideally be 0.

* `RX Bytes, TX Bytes`: total amount of data received, transmitted.


### arp

command to show the ARP (Address Resolution Protocol) cache.

ARP cache is a collection of ARP entries (IP to MAC) for efficiently communicate with the IP address.

```bash
# show all entries
arp -a 
# verbose
arp -v
```


### route, ip, netstat

* show routing table

  ```bash
  # show routing table
  route
  netstat -r
  # replace domain name for numerical IP address
  route -n
  netstat -nr
  # explain the table
  ip route show
  ```

  ```bash
  ### output example of `route -n`
  Kernel IP routing table
  Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
  0.0.0.0         222.29.2.1      0.0.0.0         UG    0      0        0 enp129s0f0 # default gateway
  169.254.0.0     0.0.0.0         255.255.0.0     U     1000   0        0 enp129s0f0
  172.17.0.0      0.0.0.0         255.255.0.0     U     0      0        0 docker0
  222.29.2.0      0.0.0.0         255.255.255.0   U     0      0        0 enp129s0f0
  
  # 0.0.0.0 or * means None (not set)
  # genmask: netmask
  # flags: U = up, G = gateway
  
  # BTW, equivalent in win10 powershell is `route PRINT`
  # 10.1.0.1: default router IP reserved for wireless routers.
  
  ### output example of `ip route show`
  default via 222.29.2.1 dev enp129s0f0 onlink
  169.254.0.0/16 dev enp129s0f0  scope link  metric 1000
  172.17.0.0/16 dev docker0  proto kernel  scope link  src 172.17.0.1 linkdown
  222.29.2.0/24 dev enp129s0f0  proto kernel  scope link  src 222.29.2.65
  ```

  
* modifying route:

  ```bash
  # add a new route
  ip route add <target/mask> via <next hop> dev <interface>
  
  # delete route
  ip route del <target/mask> via <next hop> dev <interface>
  
  # add reject route
  route add -host <target> reject # `ping <target>` will show unreachable
  
  ```

  
### subnet mask

Each host is configured with a unique IP address and subnet mask. 

The IP address is divided by the subnet mask to 2 parts: **Network ID + Host ID**

```bash
# example
inet addr:222.29.2.44  Mask:255.255.255.0

# CIDR format is 222.29.2.44/24 
# 24 ones = 11111111.11111111.11111111.00000000 = 255.255.255.0

# network id = 222.29.2.0
# host id    = 0.0.0.44
```


### what router does ?

When presented with a packet bound for an IP address

the router needs to determine which of its network interfaces will best get that packet closer to its destination.


### default gateway

The node that serves as the default router when no other route specification matches the destination IP address.

```bash
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         222.29.2.1      0.0.0.0         UG    0      0        0 enp129s0f0 # default gateway
169.254.0.0     0.0.0.0         255.255.0.0     U     1000   0        0 enp129s0f0
172.17.0.0      0.0.0.0         255.255.0.0     U     0      0        0 docker0
222.29.2.0      0.0.0.0         255.255.255.0   U     0      0        0 enp129s0f0

# 0.0.0.0, or * means ANY IP.
# 169.254.0.0 or link-local is for communication within the network segment.
```


### traceroute

install by `sudo apt install traceroute`. 

The name is not `traceroute6` !

```bash
# trace route to an IP
traceroute bilibili.com
traceroute 8.8.8.8 # google DNS

# BTW, equivalent in win10 powershell is `tracert`
```


### iptables

firewall.

```bash
# list ip tables 
iptables -L
```


