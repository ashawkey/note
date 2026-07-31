# docker hub

Docker hosts images in a **registry** server. 

Docker hub is the official site for hosting images pushed by all users.

### publish image

```bash
# build the image locally.
docker built -t <name>:<tag> .

# login to a registry
docker login -u <user> 

# create a remote repository in dockerhub.
# e.g., the repo is <user>/<name>

# tag local image
docker tag <name>:<tag> <user>/<name>:<tag>

# push
docker push <user>/<name>:<tag>

# logout
docker logout
```

