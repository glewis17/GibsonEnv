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

RUN echo "export DISPLAY=:0" >> ~/.bashrc

WORKDIR /root

RUN apt-get install -y git build-essential cmake libopenmpi-dev 
		
RUN apt-get install -y zlib1g-dev

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

RUN  apt-get install -y vim wget unzip tmux iputils-ping

RUN  apt-get install -y libzmq3-dev

ADD  . /root/mount/gibson
WORKDIR /root/mount/gibson

# Run installs for py35
ENV PATH /miniconda/envs/py35/bin:$PATH

RUN pip install --upgrade pip
RUN pip install pyzmq

ENV PATH $PATH_PRE

ENV QT_X11_NO_MITSHM 1

