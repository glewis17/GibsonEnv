# gibson graphical sample provided with the CUDA toolkit.

# docker build -t gibson .
# docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson

FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/cuda/samples/5_Simulations/nbody

RUN make

#CMD ./nbody

RUN apt-get update && apt-get install -y curl lsb-release

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
ENV PATH_PRE $PATH
RUN conda update -y conda
RUN conda create -y -n py35 python=3.5 
RUN conda create -y -n py27 python=2.7 

# Install ROS stuff for python 2.7
RUN echo "/opt/ros/kinetic/lib/python2.7/dist-packages\n/usr/lib/python2.7/dist-packages" > /miniconda/envs/py27/lib/python2.7/site-packages/ros.pth
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-get update
RUN apt-get install -y ros-kinetic-desktop-full
RUN rosdep init
RUN rosdep update
RUN apt-get install -y ros-kinetic-turtlebot ros-kinetic-turtlebot-apps ros-kinetic-turtlebot-interactions ros-kinetic-turtlebot-simulator ros-kinetic-kobuki-ftdi

ENV ROS_MASTER_URI http://171.64.70.117:11311
ENV TURTLEBOT_NAME turtlebot
ENV TURTLEBOT_3D_SENSOR kinect

WORKDIR /root

RUN apt-get install -y git build-essential cmake libopenmpi-dev 
		
RUN apt-get install -y zlib1g-dev

RUN git clone https://github.com/fxia22/baselines.git&& \
	pip install -e baselines

RUN apt-get update && apt-get install -y \
		libglew-dev \
		libglm-dev \
		libassimp-dev \
		xorg-dev \
		libglu1-mesa-dev \
		libboost-dev \
		mesa-common-dev \
		freeglut3-dev \
		libopenmpi-dev \
		cmake \
		golang \
		libjpeg-turbo8-dev \
		wmctrl \ 
		xdotool \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/cache/apk/*

RUN  apt-get install -y vim wget unzip tmux

RUN  apt-get install -y libzmq3-dev

ADD  . /root/mount/gibson
WORKDIR /root/mount/gibson

# Run installs for py27
RUN conda install -n py27 numpy pyyaml

ENV PATH /miniconda/envs/py27/bin:$PATH

RUN bash build.sh build_local
RUN pip install --upgrade pip==9.0.3
RUN pip install pyzmq
RUN pip install -e .

ENV PATH $PATH_PRE

# Run installs for py35
ENV PATH /miniconda/envs/py35/bin:$PATH

RUN pip install pyzmq

ENV PATH $PATH_PRE

RUN echo 'source /opt/ros/kinetic/setup.bash' >> ~/.bashrc
RUN echo 'unset PYTHONPATH' >> ~/.bashrc

RUN mkdir -p ~/catkin_ws/src
RUN ln -s $PWD/examples/ros/gibson-ros/ ~/catkin_ws/src/

ENV QT_X11_NO_MITSHM 1

