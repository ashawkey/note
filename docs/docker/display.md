# docker with display

Mostly need to build with `libglvnd` support.

```Dockerfile
FROM ubuntu:18.04

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# custom command
RUN apt-get update \
  && apt-get install -y -qq glmark2

CMD glmark2
```


> NO success yet, to be continued...