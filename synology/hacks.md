# hacks

Mainly from [here](https://zhuanlan.zhihu.com/p/134169554).

* stop auto update

  ```bash
  vi /etc/hosts
  
  # add the following line
  127.0.0.1 update.synology.com 
  ```

* remove auto update red point by replacing icons.

  ```bash
  cp /usr/syno/synoman/webman/modules/AdminCenter/images/default/1x/badge_control_panel.png /usr/syno/synoman/webman/modules/AdminCenter/images/default/1x/badge_control_panel.png.bak  
  cp /usr/syno/synoman/webman/modules/AdminCenter/images/default/2x/badge_control_panel.png /usr/syno/synoman/webman/modules/AdminCenter/images/default/2x/badge_control_panel.png.bak  
  wget -O /usr/syno/synoman/webman/modules/AdminCenter/images/default/1x/badge_control_panel.png https://static.iots.vip/transparent.png
  wget -O /usr/syno/synoman/webman/modules/AdminCenter/images/default/2x/badge_control_panel.png https://static.iots.vip/transparent.png
  
  cp /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_notification_num.png /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_notification_num.png.bak
  cp /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_notification_num.png /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_notification_num.png.bak
  wget -O /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_notification_num.png https://static.iots.vip/transparent.png
  wget -O /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_notification_num.png https://static.iots.vip/transparent.png
  
  cp /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_badge_num.png /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_badge_num.png.bak 
  cp /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_badge_num.png /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_badge_num.png.bak 
  wget -O /usr/syno/synoman/synoSDSjslib/images/default/1x/dsm5_badge_num.png https://static.iots.vip/transparent.png
  wget -O /usr/syno/synoman/synoSDSjslib/images/default/2x/dsm5_badge_num.png https://static.iots.vip/transparent.png
  ```

* CPU info

  ```bash
  wget -N --no-check-certificate http://static.iots.vip/sh/ch_cpuinfo.sh && sh ch_cpuinfo.sh
  ```

* optimize hard disk IO by moving log to memory

  ```bash
  sed -i 's/var\/log/dev\/shm/' /etc.defaults/syslog-ng/patterndb.d/scemd.conf
  ```

* `invalid location` (无效的位置) error on DSM6 when adding new package sources. from [here](https://www.ruoyer.com/dsm_crt.html) and [here](https://github.com/SynoCommunity/spksrc/issues/4897#issuecomment-961876276).

  caused by expiration of let's encrypt certificate.

  ```bash
  # backup
  sudo mv /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt.bak
  # install new
  sudo curl -Lko /etc/ssl/certs/ca-certificates.crt https://curl.se/ca/cacert.pem
  ```

  

