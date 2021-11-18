# filebrowser

[website](https://filebrowser.org/)



### install

* official repo:

  search `filebrowser` in the Apps, and simply mount the targeted directory.

* enhanced version:

  https://hub.docker.com/r/80x86/filebrowser

  need to manually add docker container.

  ```bash
  repo: 80x86/filebrowser:2.9.4-amd64
  port: 8082:8082
  path: /config:/mnt/user/appdata/fbenhanced
  path: /myfiles:/mnt/user/
  device: /dev/dri
  ```

  



### usage

Default account/password is `admin/admin`.

