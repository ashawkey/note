# disk operation

```
|-----------------------disk---------------------|
|-----partition------|---------partition---------|
|-----filesystem-----|---------filesystem--------|

ref: https://stackoverflow.com/questions/24429949/device-vs-partition-vs-file-system-vs-volume-how-do-these-concepts-relate-to-ea
```


### Partition Table Type

A data structure that provides basic information for the OS about the partition of the hard disk.

* MBR: Master Boot Record.

  Used with the legacy BIOS firmware.

  An older standard. (only support < 2T disk)

* **GPT**: GUID Partition Table

  Used with the EFI firmware. (not compatible with BIOS)

  Current default.

  GPT contains a fake MBR at the beginning of its space. This MBR shows the drive as being a single MBR partition to cope with tools which do not recognize GPT.

### Filesystem

A way of storing data inside the partitions.

* EXT / EXT2 / EXT3

  older version of EXT.

* **EXT4**

  current default for Linux, Android.

  support both Windows and Linux. support < 1EB = 2^20 TB.

* FAT

  much older, for Windows. FAT16 < 4GB, FAT32 < 32 GB.

* **NTFS**

  current default for Windows.

* VFAT

  support both Windows and Linux, usually for swap and recovery.


### check disk state and mount point

* `fdisk`: check all physically loaded disks

  ```bash
  $ sudo fdisk -l
  Disk /dev/sda: 1.8 TiB, 2000398934016 bytes, 3907029168 sectors
  Units: sectors of 1 * 512 = 512 bytes
  Sector size (logical/physical): 512 bytes / 4096 bytes
  I/O size (minimum/optimal): 4096 bytes / 4096 bytes
  Disklabel type: gpt
  Disk identifier: 943FD3C6-B398-4D2E-8A66-58991158B3A2
  
  Device          Start        End    Sectors  Size Type
  /dev/sda1        2048    1050623    1048576  512M EFI System
  /dev/sda2     1050624 3905028095 3903977472  1.8T Linux filesystem
  /dev/sda3  3905028096 3907028991    2000896  977M Linux swap
  
  
  Disk /dev/sdb: 953.9 GiB, 1024209543168 bytes, 2000409264 sectors
  Units: sectors of 1 * 512 = 512 bytes
  Sector size (logical/physical): 512 bytes / 512 bytes
  I/O size (minimum/optimal): 512 bytes / 512 bytes
  
  
  Disk /dev/sdc: 931.5 GiB, 1000204886016 bytes, 1953525168 sectors
  Units: sectors of 1 * 512 = 512 bytes
  Sector size (logical/physical): 512 bytes / 512 bytes
  I/O size (minimum/optimal): 512 bytes / 512 bytes
  
  
  Disk /dev/sdd: 931.5 GiB, 1000204886016 bytes, 1953525168 sectors
  Units: sectors of 1 * 512 = 512 bytes
  Sector size (logical/physical): 512 bytes / 512 bytes
  I/O size (minimum/optimal): 512 bytes / 512 bytes
  ```

  **Linux Filesystem** is where the root  `/` is mounted.

  **EFI System** (Extensible Firmware Interface) is used to boot the system. (UEFI firmware loads the files on the EFI System)

* `mount` check the mount 

  ```bash
  $ mount
  sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)
  proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)
  udev on /dev type devtmpfs (rw,nosuid,relatime,size=131978096k,nr_inodes=32994524,mode=755)
  devpts on /dev/pts type devpts (rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=000)
  tmpfs on /run type tmpfs (rw,nosuid,noexec,relatime,size=26402968k,mode=755)
  /dev/sda2 on / type ext4 (rw,relatime,errors=remount-ro,data=ordered)
  securityfs on /sys/kernel/security type securityfs (rw,nosuid,nodev,noexec,relatime)
  tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)
  tmpfs on /run/lock type tmpfs (rw,nosuid,nodev,noexec,relatime,size=5120k)
  tmpfs on /sys/fs/cgroup type tmpfs (ro,nosuid,nodev,noexec,mode=755)
  cgroup on /sys/fs/cgroup/systemd type cgroup (rw,nosuid,nodev,noexec,relatime,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd)
  pstore on /sys/fs/pstore type pstore (rw,nosuid,nodev,noexec,relatime)
  efivarfs on /sys/firmware/efi/efivars type efivarfs (rw,nosuid,nodev,noexec,relatime)
  cgroup on /sys/fs/cgroup/freezer type cgroup (rw,nosuid,nodev,noexec,relatime,freezer)
  cgroup on /sys/fs/cgroup/rdma type cgroup (rw,nosuid,nodev,noexec,relatime,rdma)
  cgroup on /sys/fs/cgroup/devices type cgroup (rw,nosuid,nodev,noexec,relatime,devices)
  cgroup on /sys/fs/cgroup/cpu,cpuacct type cgroup (rw,nosuid,nodev,noexec,relatime,cpu,cpuacct)
  cgroup on /sys/fs/cgroup/blkio type cgroup (rw,nosuid,nodev,noexec,relatime,blkio)
  cgroup on /sys/fs/cgroup/net_cls,net_prio type cgroup (rw,nosuid,nodev,noexec,relatime,net_cls,net_prio)
  cgroup on /sys/fs/cgroup/hugetlb type cgroup (rw,nosuid,nodev,noexec,relatime,hugetlb)
  cgroup on /sys/fs/cgroup/cpuset type cgroup (rw,nosuid,nodev,noexec,relatime,cpuset)
  cgroup on /sys/fs/cgroup/perf_event type cgroup (rw,nosuid,nodev,noexec,relatime,perf_event)
  cgroup on /sys/fs/cgroup/memory type cgroup (rw,nosuid,nodev,noexec,relatime,memory)
  cgroup on /sys/fs/cgroup/pids type cgroup (rw,nosuid,nodev,noexec,relatime,pids)
  systemd-1 on /proc/sys/fs/binfmt_misc type autofs (rw,relatime,fd=35,pgrp=1,timeout=0,minproto=5,maxproto=5,direct,pipe_ino=55306)
  debugfs on /sys/kernel/debug type debugfs (rw,relatime)
  mqueue on /dev/mqueue type mqueue (rw,relatime)
  hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=2M)
  fusectl on /sys/fs/fuse/connections type fusectl (rw,relatime)
  configfs on /sys/kernel/config type configfs (rw,relatime)
  /dev/sdb on /data2 type ext4 (rw,relatime,data=ordered)
  /dev/sdc on /data3 type ext4 (rw,relatime,data=ordered)
  /dev/sdd on /data4 type ext4 (rw,relatime,data=ordered)
  /dev/sda1 on /boot/efi type vfat (rw,relatime,fmask=0077,dmask=0077,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro)
  tmpfs on /run/user/1002 type tmpfs (rw,nosuid,nodev,relatime,size=26402968k,mode=700,uid=1002,gid=1002)
  
  ```

  Note the physical disk lines:

  ```
  /dev/sda2 on / type ext4 (rw,relatime,errors=remount-ro,data=ordered)
  /dev/sdb on /data2 type ext4 (rw,relatime,data=ordered)
  /dev/sdc on /data3 type ext4 (rw,relatime,data=ordered)
  /dev/sdd on /data4 type ext4 (rw,relatime,data=ordered)
  /dev/sda1 on /boot/efi type vfat 
  ```

* `df`  simplified version.

  ```bash
  $ df -h
  Filesystem      Size  Used Avail Use% Mounted on
  udev            126G     0  126G   0% /dev
  tmpfs            26G   19M   26G   1% /run
  /dev/sda2       1.8T  1.5T  214G  88% /
  tmpfs           126G     0  126G   0% /dev/shm
  tmpfs           5.0M  4.0K  5.0M   1% /run/lock
  tmpfs           126G     0  126G   0% /sys/fs/cgroup
  /dev/sdb        939G  562G  330G  64% /data2
  /dev/sdc        917G  729G  142G  84% /data3
  /dev/sdd        917G   72M  871G   1% /data4
  /dev/sda1       511M  3.7M  508M   1% /boot/efi
  tmpfs            26G     0   26G   0% /run/user/1002
  ```

* `lsblk` another simplified version.

  ```bash
  $ sudo lsblk
  NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
  sdd      8:48   0 931.5G  0 disk /data4
  sdb      8:16   0 953.9G  0 disk /data2
  sdc      8:32   0 931.5G  0 disk /data3
  sda      8:0    0   1.8T  0 disk
  ├─sda2   8:2    0   1.8T  0 part /
  ├─sda3   8:3    0   977M  0 part [SWAP]
  └─sda1   8:1    0   512M  0 part /boot/efi
  ```

  To check if the disk is SSD or HDD (ROTAtional):

  ```bash
  $ lsblk -d -o name,rota
  NAME ROTA
  sda     0
  sdb     0
  sdc     1
  ```

  where ROTA == 1 means HDD, and ROTA == 0 means SSD.

* `findmnt` specialized tool

  ```bash
  $ findmnt
  TARGET                                SOURCE     FSTYPE     OPTIONS
  /                                     /dev/sda2  ext4       rw,relatime,errors=remount-ro,data=ordered
  ├─/sys                                sysfs      sysfs      rw,nosuid,nodev,noexec,relatime
  │ ├─/sys/kernel/security              securityfs securityfs rw,nosuid,nodev,noexec,relatime
  │ ├─/sys/fs/cgroup                    tmpfs      tmpfs      ro,nosuid,nodev,noexec,mode=755
  │ │ ├─/sys/fs/cgroup/systemd          cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd
  │ │ ├─/sys/fs/cgroup/freezer          cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,freezer
  │ │ ├─/sys/fs/cgroup/rdma             cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,rdma
  │ │ ├─/sys/fs/cgroup/devices          cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,devices
  │ │ ├─/sys/fs/cgroup/cpu,cpuacct      cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,cpu,cpuacct
  │ │ ├─/sys/fs/cgroup/blkio            cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,blkio
  │ │ ├─/sys/fs/cgroup/net_cls,net_prio cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,net_cls,net_prio
  │ │ ├─/sys/fs/cgroup/hugetlb          cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,hugetlb
  │ │ ├─/sys/fs/cgroup/cpuset           cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,cpuset
  │ │ ├─/sys/fs/cgroup/perf_event       cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,perf_event
  │ │ ├─/sys/fs/cgroup/memory           cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,memory
  │ │ └─/sys/fs/cgroup/pids             cgroup     cgroup     rw,nosuid,nodev,noexec,relatime,pids
  │ ├─/sys/fs/pstore                    pstore     pstore     rw,nosuid,nodev,noexec,relatime
  │ ├─/sys/firmware/efi/efivars         efivarfs   efivarfs   rw,nosuid,nodev,noexec,relatime
  │ ├─/sys/kernel/debug                 debugfs    debugfs    rw,relatime
  │ ├─/sys/fs/fuse/connections          fusectl    fusectl    rw,relatime
  │ └─/sys/kernel/config                configfs   configfs   rw,relatime
  ├─/proc                               proc       proc       rw,nosuid,nodev,noexec,relatime
  │ └─/proc/sys/fs/binfmt_misc          systemd-1  autofs     rw,relatime,fd=35,pgrp=1,timeout=0,minproto=5,maxproto=5,direct,pipe_ino=55306
  ├─/dev                                udev       devtmpfs   rw,nosuid,relatime,size=131978096k,nr_inodes=32994524,mode=755
  │ ├─/dev/pts                          devpts     devpts     rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=000
  │ ├─/dev/shm                          tmpfs      tmpfs      rw,nosuid,nodev
  │ ├─/dev/mqueue                       mqueue     mqueue     rw,relatime
  │ └─/dev/hugepages                    hugetlbfs  hugetlbfs  rw,relatime,pagesize=2M
  ├─/run                                tmpfs      tmpfs      rw,nosuid,noexec,relatime,size=26402968k,mode=755
  │ ├─/run/lock                         tmpfs      tmpfs      rw,nosuid,nodev,noexec,relatime,size=5120k
  │ └─/run/user/1002                    tmpfs      tmpfs      rw,nosuid,nodev,relatime,size=26402968k,mode=700,uid=1002,gid=1002
  ├─/data2                              /dev/sdb   ext4       rw,relatime,data=ordered
  ├─/data3                              /dev/sdc   ext4       rw,relatime,data=ordered
  ├─/data4                              /dev/sdd   ext4       rw,relatime,data=ordered
  └─/boot/efi                           /dev/sda1  vfat       rw,relatime,fmask=0077,dmask=0077,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro
  
  $ findmnt /
  TARGET SOURCE    FSTYPE OPTIONS
  /      /dev/sda2 ext4   rw,relatime,errors=remount-ro,data=ordered
  ```


### Add a new disk

* Plug in the hardware

* Check is it detected

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
  
  # <disk> <mount> <filesystem> <option> <dump, should be 0> <pass, root disk shoule be 1, other disk should be 2>
  
  /dev/sdb /data2 ext4 defaults 0 2
  ```
  
* check auto-mount

  ```bash
  sudo mount -av
  ```

  


### testdisk

TestDisk can undelete

- [files and directory from FAT12, FAT16 and FAT32 filesystem](https://www.cgsecurity.org/wiki/TestDisk:_undelete_file_for_FAT),
- [files from ext2 filesystem](https://www.cgsecurity.org/wiki/TestDisk:_undelete_file_for_ext2),
- [files from NTFS partition](https://www.cgsecurity.org/wiki/Undelete_files_from_NTFS_with_TestDisk) since version [6.11](https://www.cgsecurity.org/wiki/TestDisk_6.11_Release).

If it doesn't work or for other filesystem! (which means the most used ext4 cannot use it.)


### extundelete

Need to unmount the device.

Usually we need a live-USB boot. (very difficult, not recommended)


### if the process is not killed (the only practical way)

> ref: https://www.jianshu.com/p/662293f12a47

```bash
# find the pid that occupies the deleted file.
lsof | grep <deletedfilename>
# copy the proc, then you can find it in /backup
cp /proc/<pid>/fd /backup
```

```
$ ls /proc/<pid>/fd
lrwx------ 1 root root 64 1月  18 22:21 0 -> /dev/pts/0
l-wx------ 1 root root 64 1月  18 22:21 1 -> /root/deletefile.txt (deleted)
lrwx------ 1 root root 64 1月  18 22:21 2 -> /dev/pts/0
```

