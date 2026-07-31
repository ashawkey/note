# Docker Tricks

## Bake an image from a container

Sometimes you need really strange environments which are hard to build from Dockerfile.
You can choose another risky yet easy way: **manually tune the environment in a running container, then bake it to an image**.

```bash
# run a container from a base image
docker run -it --name my_container base_image

# enter the container
docker exec -it my_container /bin/bash

# modify the environment
python whatever_script.py # download model weights to god-know-where folder, and they will be kept in the image!

# exit the container
exit

# bake the container to an image
docker commit my_container my_image

# now you can start a new container from the image, and all you did is baked!
docker run -it my_image /bin/bash
```

This image cannot be reproduced by a Dockerfile, but it is easier to debug.