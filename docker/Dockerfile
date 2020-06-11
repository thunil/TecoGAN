#from ubuntu:18.04
#from nvidia/cuda:10.2-base
from tensorflow/tensorflow:1.8.0-gpu-py3

run apt-get update && apt-get install -y \
git \
python3-pip

workdir /
run git clone https://github.com/thunil/TecoGAN.git
workdir /TecoGAN
#run pip3 install --ignore-installed --upgrade tensorflow-gpu==1.8.0 # or tensorflow
run pip3 install --upgrade setuptools
run pip3 install --upgrade pip
#run pip3 install \
#scikit-image==0.13.0 \
#matplotlib==1.5.1 \
#numpy==1.14.3 \
#keras==2.1.2

run pip3 install \
numpy==1.14.3 \
scipy==1.0.1 \
scikit-image==0.13.0 \
matplotlib==1.5.1 \
pandas==0.23.1 \
Keras==2.1.2 \
torch==0.4.0 \
torchvision==0.2.1 \
opencv-python \
ipython==7.4.0

#run pip3 install -r requirements.txt

run apt-get update && apt-get install -y \
wget \
unzip

run python3 runGan.py 0

run apt-get update && apt-get install -y \
libsm6 \
libxrender1 \
libxext6

#run pip uninstall -y numpy && pip install numpy==1.14.3
