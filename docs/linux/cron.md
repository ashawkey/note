# crontab

### config files

```bash
# system cron jobs
/etc/crontab

/var/spool/cron.d/*

/ect/cron.d/*

# put a shell script under these folders will automatically do the job:
/ect/cron.hourly/*
/ect/cron.daily/*
/ect/cron.weekly/*
/ect/cron.monthly/*
```

Config formats:

```bash
# each field supports single number, lists, intervals:
# minute hour day-of-month month day-of-week command

30 17 * * 1 cmd # every Monday 17:30
0,30 17 * * 1 cmd # every Monday 17:00 and 17:30
*/15 * * * * cmd # every 15 minutes

# note day-of-week: 0=Sunday, 1=Monday, ... , 6=Saturday.

# special strings:
@reboot cmd # run once at startup
@hourly cmd # 0 * * * *
@daily cmd # 0 0 * * *
@weekly cmd # 0 0 * * 0
@monthly cmd # 0 0 1 * *
@yearly cmd # 0 0 1 1 *
```


### cron deamon

```bash
systemctl status cron
systemctl stop cron
systemctl restart cron
```


### cron command

```bash
crontab <file> # use file to readin jobs.
crontab -e # edit per-user cron jobs
crontab -l # list cron jobs
crontab -l -u <user>
```


### log files

```bash
# first place to check
/var/log/syslog

# for auth
/var/log/auth.log
```


### anacron

设想这样一个场景，Linux 服务器会在周末关机两天，但是设定的定时任务大多在周日早上进行，但在这个时间点，服务器又处于关机状态，导致系统很多定时任务无法运行。

又比如，我们需要在凌晨 5 点 05 分执行系统的日志备份，但 Linux 服务器不是 24 小时开机的，在晚上需要关机，白天上班之后才会再次开机，在这个定时任务的执行时间我们的服务器刚好没有开机，那么这个定时任务就不会执行了。anacron 就是用来解决这个问题的。

anacron 会以 1 天、1周、1月作为检测周期，判断是否有定时任务在关机之后没有执行。如果有这样的任务，那么 anacron 会在特定的时间重新执行这些定时任务。

```bash
# /etc/anacrontab: configuration file for anacron
# See anacron(8) and anacrontab(5) for details.
SHELL=/bin/sh
PATH=/sbin:/bin:/usr/sbin:/usr/bin MAILTO=root
#前面的内容和/etc/crontab类似
#the maximal random delay added to the base delay of the jobs
RANDOM_DELAY=45
#最大随机廷迟
#the jobs will be started during the following hours only
START_H0URS_RANGE=3-22
#fanacron的执行时间范围是3:00~22:00
#period in days delay in minutes job-identifier command
1 5 cron.daily nice run-parts /etc/cron.daily
#每天开机 5 分钟后就检查 /etc/cron.daily 目录内的文件是否被执行，如果今天没有被执行，那就执行
7 25 cron.weekly nice run-parts /etc/cron.weekly
#每隔 7 天开机后 25 分钟检查 /etc/cron.weekly 目录内的文件是否被执行，如果一周内没有被执行，就会执行
©monthly 45 cron.monthly nice run-parts /etc/cron.monthly
#每隔一个月开机后 45 分钟检查 /etc/cron.monthly 目录内的文件是否被执行，如果一个月内没有被执行，那就执行 
```

