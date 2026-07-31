# pm2

A daemon manager for `node.js` backend servers.

### install

```bash
npm install -g pm2
```


### usage

```bash
# start a server in background
pm2 start [--name <name>] [--log <path>] app.js

# start using bash
pm2 start run.sh

# with env variables
PORT=8088 pm2 start app.js

# list
pm2 l|ls|list|status

# attach to output logs
pm2 logs # all
pm2 logs <name/id>

# manage log in a dashboard
pm2 monit

# manage
pm2 restart <name/id> # use `all` to act on all apps.
pm2 stop <name/id>
pm2 delete <name/id>
```

