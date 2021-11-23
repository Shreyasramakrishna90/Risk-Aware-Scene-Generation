#FROM ubuntu:18.04
FROM nvidia/cudagl:10.1-devel-ubuntu18.04
RUN mkdir -p carla-challange
RUN mkdir -p routes
RUN mkdir -p simulation-data
WORKDIR carla-challange
RUN apt-get update && apt-get install apt-transport-https

#install pip
RUN apt-get install -y python3
RUN apt-get update && apt-get install -y  python3-pip
RUN apt-get install -y python3.7
RUN python3.7 -m pip install pip
RUN python3.7 -m pip install --upgrade pip
RUN apt-get install -y vim
#RUN apt-get install -y python3.7-lxml
COPY requirements.txt requirements.txt
RUN python3.7 -m pip install -r requirements.txt
RUN apt install -y libtiff5-dev
RUN apt-get install -y libjpeg-turbo8
RUN apt-get install -y libpng16-16
RUN apt-get install -y libsm6 libxrender1 libfontconfig1
RUN apt-get install -y libxext6 libgl1-mesa-glx
#RUN apt install unzip
#COPY carla-challange/libs libs
#RUN tar -xzvf libs/carla-0.9.9-py3.7-linux-x86_64.tar.xz -C libs/carla-0.9.9-py3.7-linux-x86_64
#RUN unzip -d libs/carla-0.9.9-py3.7-linux-x86_64 libs/carla-0.9.9-py3.7-linux-x86_64.zip
#COPY libs/carla-0.9.9-py3.7-linux-x86_64 libs/setup.py
#RUN python3.7 -m pip install -e libs/carla-0.9.9-py3.7-linux-x86_64
ENV TZ=US/Central
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y python3.7-tk
RUN useradd -p vandy -ms /bin/bash  carla
RUN apt-get -y install sudo
RUN  usermod -aG sudo carla
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER carla
