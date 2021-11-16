# frp docker



### client side by docker

* create `/boot/config/frp/frpc.ini` and write:

  ```ini
  [common]
  server_addr = [server]
  server_port = 7000
  token = [token]
  
  [ssh]
  type = tcp
  local_ip = 127.0.0.1
  local_port = 22
  remote_port = 2222
  
  [webui]
  type = http
  local_ip = 127.0.0.1
  local_port = 80
  custom_domains = nas.kiui.moe
  
  [jellyfin]
  type = http
  local_ip = 127.0.0.1
  local_port = 8096
  custom_domains = jellyfin.kiui.moe
  
  ```
  
  

* pull and run

  ```bash
  docker run --restart=always --network host -d -v /boot/config/frp/frpc.ini:/etc/frp/frpc.ini --name frpc snowdreamtech/frpc
  ```

  

 

