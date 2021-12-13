# docker-compose

### install

Follow https://docs.docker.com/compose/install/

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```



### usage

Run multiple containers (service) within one config (project).

Create an empty folder, and create  `docker-compose.yml`:

```yaml
version: '3'
services:
  web:
    build: . # build image from current dir
    ports:
     - "5000:5000"
  redis:
    image: "redis:alpine" # use image
```

Run it:

```bash
# run in foreground
docker-compose up

# run in background (detached)
docker-compuse up -d

# check status
docker-compose ps

# stop
docker-compose down
```

