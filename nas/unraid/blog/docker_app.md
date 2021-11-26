# Plugins/Apps推荐

### Plugins

从Apps中直接搜索即可安装。

* Dynamix System Buttons

  体验优化，增加一个关机、重启、操作阵列的按钮。

* Dynamix System Information

  体验优化，显示详细系统信息。

* Dynamix System Statistics

  体验优化，显示随时间的系统资源统计。

* Dynamix System Temperature

  体验优化，显示主板、CPU温度以及风扇转速。

* Nerd Tools

  在终端安装各种常用命令，如`vim, tmux, python`。

  由于需要下载很多东西，请配置好代理后再使用。

* compose.manager

  在终端安装`docker-compose`。



### Docker Apps

可以从Apps中选择模板安装，如果自己创建模板则需要一定的Docker知识。

这里均以docker仓库的名称表示App。

* `filebrowser/filebrowser`

  文件管理系统。优美的Web UI，补全NAS体验的重要环节，强烈推荐！

* `deluan/navidrome`

  音乐库管理系统。

* `lscr.io/linuxserver/jellyfin:latest`

  视频库管理系统。开启硬件解码的方法：

  * 终端运行`modprobe i915`，并在`/boot/config/go`中添加这个命令以持久化。
  * 容器设置中添加设备`/dev/dri`。

* `hotio/qbittorrent:release`

  Torrent下载器。





