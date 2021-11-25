#!/bin/bash

#pull carla image
if [ ! -d CARLA_0.9.9 ]
then
  echo $OSTYPE
  if [[ "$OSTYPE" == "linux-gnu"* ]];
  then
    echo "CARLA Simulator package not found. Downloading the simulator package"
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.tar.gz
    mkdir CARLA_0.9.9
    tar -xzvf CARLA_0.9.9.tar.gz -C CARLA_0.9.9
    rm -rf CARLA_0.9.9.tar.gz
  else
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/CARLA_0.9.9.zip
    mkdir CARLA_0.9.9
    unzip -d CARLA_0.9.9 CARLA_0.9.9.zip
    rm -rf CARLA_0.9.9.zip
  fi
else
  echo "CARLA Simulator package found"

fi

echo $PWD/carla-challenge/carla_project/model
#pull trained LEC weights
if [ ! -d $PWD/carla-challenge/carla_project/model ]
then
  echo "LEC model already exists"
  unzip -d $PWD/carla-challenge/carla_project/model Eaq1ptU-YJJPrqmEYUK_dx8Bad2KqhVQZJkKwngWnuMWRA?e=U3dtyf.zip
  #rm -rf model.zip
else
  echo "Pulling the trained LEC weights"
  #wget "https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/Eaq1ptU-YJJPrqmEYUK_dx8Bad2KqhVQZJkKwngWnuMWRA?e=U3dtyf&download=1"
  #unzip -d carla-challenge/carla_project model.zip
  #rm -rf model.zip

fi