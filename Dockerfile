FROM ubuntu:16.04
LABEL maintainer="smirk.cao@gmail.com"
## TBD
# python3

## build docker image
# $ xhost +
# $ mkdir docker
# $ cp Dockerfile docker/.
# $ cd docker && sudo docker build -t smirk/opencv .

## run docker
# $ cd .. && sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v `pwd`:/home/smirk smirk/opencv:latest


RUN apt-get update
# dependencies
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
   libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev \
   libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

RUN apt-get install -y unzip vim wget
ENV OPENCV_VERSION="3.4.2"

RUN cd \
   && wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
   && unzip ${OPENCV_VERSION}.zip \
   && mv opencv-${OPENCV_VERSION} opencv \
   && rm ${OPENCV_VERSION}.zip

RUN cd \
   && wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
   && unzip ${OPENCV_VERSION}.zip \
   && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \
   && rm ${OPENCV_VERSION}.zip

RUN cd \
   && cd opencv \
   && mkdir build \
   && cd build \
   && cmake \
   -D CMAKE_BUILD_TYPE=RELEASE \
   -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION}/  \
   -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
   -D WITH_TBB=ON \
   -D WITH_V4L=ON \
   -D WITH_GTK=ON \
   -D WITH_OPENGL=ON \
   -D WITH_CUDA=OFF \
   -D WITH_CUBLAS=OFF \
   -D BUILD_EXAMPLES=OFF \
   -D INSTALL_PYTHON_EXAMPLES=OFF \
   -D INSTALL_C_EXAMPLES=OFF \
   .. \
   && make -j8 \
   && make install \
   && ldconfig \
   && cd

RUN cd \
   && git clone https://github.com/SmirkCao/LEGO_detection.git \
   && cd AdrianKaehler/CPP \
   && mkdir build \
   && cd build \
   && cmake .. \
   && make