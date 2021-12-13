### wget

```bash
# download 
wget https://example.com/test.txt

# continue download
wget -c https://example.com/test.txt
```



### cURL (client URL)

```bash
# GET and print response
curl https://example.com

# GET and save response to file, == wget
curl -o test.html https://example.com

# query local ip (using https://ip.sb)
curl -4 ip.sb # ipv4
curl ip.sb # ipv6
```





### aria2

