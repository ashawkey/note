# frp docker


### client side by docker

* create `/boot/config/frp/frpc.ini` and write:

  ```ini
  [common]
  server_addr = 149.28.65.26
  server_port = 7000
  token = tjx19990121
  
  ; dangerous!
  ; [ssh]
  ; type = tcp
  ; local_ip = 127.0.0.1
  ; local_port = 22
  ; remote_port = 2222
  
  ; dangerous!
  ; [webui]
  ; type = http
  ; local_ip = 127.0.0.1
  ; local_port = 80
  ; custom_domains = nas.kiui.moe
  
  [webdav]
  type = tcp
  local_ip = 127.0.0.1
  local_port = 8384
  remote_port = 8384
  
  ; [jellyfin_tcp]
  ; type = tcp
  ; local_ip = 127.0.0.1
  ; local_port = 8096
  ; remote_port = 10002
  
  [jellyfin]
  type = http
  local_ip = 127.0.0.1
  local_port = 8096
  custom_domains = jellyfin.kiui.moe
  
  ; FAILED. 
  ; [p2p_ssh]
  ; type = xtcp
  ; sk = jdafoinjasdfjk
  ; local_ip = 127.0.0.1
  ; local_port = 22
  
  ; [p2p_webui]
  ; type = xtcp
  ; sk = jdafoinjasdfjk
  ; local_ip = 127.0.0.1
  ; local_port = 80
  
  ; [p2p_jellyfin]
  ; type = xtcp
  ; sk = jdafoinjasdfjk
  ; local_ip = 127.0.0.1
  ; local_port = 8096
  
  
  ```

  

* pull and run

  ```bash
  docker run --restart=always --network host -d -v /boot/config/frp/frpc.ini:/etc/frp/frpc.ini --name frpc snowdreamtech/frpc
  ```

  

 

