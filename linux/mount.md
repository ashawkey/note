# disk management

### Check disk state

* storage

  ```bash
  df -h
  du -h --max-depth=1
  ```

* list disks

  ```bash
  sudo fdisk -l # status,partitions, ...
  sudo blkid # UUID
  ```



### Modify disk

* make / delete partitions

  ```bash
  sudo fdisk /dev/sdb
  ```

  



### Add a new disk

* Plug in the hardware

* Check is detected

  ```bash
  sudo fdisk -l 
  # should see /dev/sdb, /dev/sdc, ...
  ```

* (Optional) make partitions

  ```bash
  sudo fdisk /dev/sdb
  # >>> p
  # >>> n
  # ...
  ```

* make filesystem

  ```bash
  sudo mkfs -t ext4 /dev/sdb 
  # sudo mkfs -t ext4 /dev/sdb1 # if partitioned
  ```

* mount

  ```bash
  sudo mkdir /data2
  sudo mount /dev/sdb /data2
  ```

* set auto-mount after reboot

  ```bash
  sudo vim /etc/fstab
  # add a line:
  # /dev/sdb /data2 ext4 defaults 0 2
  ```

* check auto-mount

  ```bash
  sudo mount -av
  ```

  