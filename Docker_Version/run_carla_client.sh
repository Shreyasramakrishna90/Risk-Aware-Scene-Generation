#! /bin/bash

#run the docker container
docker run -it --rm \
    -v $(pwd)/carla-challange/:/carla-challange \
    -v $(pwd)/routes/:/carla-challange/routes \
    -v $(pwd)/images/:/carla-challange/images \
    -v $(pwd)/simulation-data/:/carla-challange/simulation-data \
    -v $(pwd)/CARLA_0.9.9/:/CARLA_0.9.9 \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    carla_challange_client:v1 bash


# --net=host
