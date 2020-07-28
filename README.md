# Semantic-segmentation-tensorflow

This repo contains a tensorflow app that applies semantic segmentation on a set of images and visualizes the extracted deep features.

## Installation
**Requirements**
- You need to have python3 installed.
- Install package requirements. <br> Run in `root` folder, 
    ~~~~
    pip install -r requirements.txt
    ~~~~
- Run in `root` folder,
     ~~~~
   ./run.py
    ~~~~
# Run with Docker

## Installation
Requirements
- You need to have [**Docker**](https://docs.docker.com/get-docker/) installed

## Run the Docker image and login to the container

Run in `root` folder,
~~~~
docker build --rm -f Dockerfile -t mytensorflow .
~~~~

Run the container,
~~~~
docker run --rm -d -p 6006:6006 -p 8888:8888 --name cv_exercise -v ${PWD}:/code mytensorflow:latest
~~~~

Login to the container,
~~~~
docker exec -it cv_exercise /bin/bash
~~~~

## Run application

Navigate to `/code` folder and run,
~~~~
./run.py
~~~~

## Alternatively run the demo using jupyter notebook
Login to the container and run,
~~~~
jupyter notebook list
~~~~
Navigate to the jupyter notebook url.



