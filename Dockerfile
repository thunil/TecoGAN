FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt-get update && apt-get install -y \
	git \
	python3-pip

WORKDIR /
RUN git clone https://github.com/thunil/TecoGAN.git
WORKDIR /TecoGAN
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip

RUN pip3 install \
	numpy==1.14.3 \
	scipy==1.0.1 \
	scikit-image==0.13.0 \
	matplotlib==1.5.1 \
	pandas==0.23.1 \
	Keras==2.1.2 \
	torch==0.4.0 \
	torchvision==0.2.1 \
	opencv-python==4.2.0.34 \
	ipython==7.4.0

RUN apt-get update && apt-get install -y \
	wget \
	unzip

RUN python3 runGan.py 0

RUN apt-get update && apt-get install -y \
	libsm6 \
	libxrender1 \
	libxext6

RUN pip3 install \
	youtube-dl

RUN apt-get update && apt-get install -y \
	ffmpeg

RUN pip3 install \
	ffmpeg==1.4

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
