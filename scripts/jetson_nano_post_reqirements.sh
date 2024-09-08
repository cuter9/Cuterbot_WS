# Make swapfile
cd
# sudo swapoff -a
# sudo fallocate -l 6G /var/swapfile
# sudo chmod 600 /var/swapfile
# sudo mkswap /var/swapfile
# sudo swapon /var/swapfile
# sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
# sleep 5m

# Install pip and some python dependencies
echo -e "\e[104m Install pip and some python dependencies \e[0m"
sudo apt update
# sudo apt install -y python3-pip python3-pil
sudo apt install -y python3-pip
sudo -H python3 -m pip install pip -U
sudo -H python3 -m pip install pillow --force-reinstall
sudo -H python3 -m pip install "Cython<3"
sudo -H python3 -m pip install --upgrade "numpy<1.19.0"

# Install jtop
echo -e "\e[100m Install jtop \e[0m"
sudo -H python3 -m pip install jetson-stats 


# Install the pre-built TensorFlow pip wheel
echo -e "\e[48;5;202m Install the pre-built TensorFlow pip wheel \e[0m"
sudo apt update
sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
# sudo apt-get install -y python3-pip
sudo -H python3 -m pip install -U testresources setuptools numpy==1.18.5 future==0.17.1 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.3.3 futures==3.1.1 "protobuf==4.21.0" pybind11
# sudo -H python3 -m pip install -U pip testresources setuptools numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 protobuf pybind11
# TF-1.15
sudo -H python3 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 'tensorflow<2'

# Install the pre-built PyTorch pip wheel
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
echo -e "\e[45m Install the pre-built PyTorch pip wheel  \e[0m"

cd $HOME/Downloads
sudo apt install -y libopenblas-base libopenmpi-dev
sudo pip3 install wget
# wget -N https://nvidia.box.com/shared/static/yr6sjswn25z7oankw8zy1roow9cy5ur1.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
wget -N https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo -H python3 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl 

# wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl -O torch-1.11.0-cp36-cp36m-linux_aarch64.whl
# sudo -H python3 -m pip install torch-1.11.0-cp36-cp36m-linux_aarch64.whl 

# Install torchvision package
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

echo -e "\e[45m Install torchvision package \e[0m"

cd $HOME/Downloads
# sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev libomp-dev ffmpeg
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
export BUILD_VERSION=0.11.1  # for torch version v1.10.0
# export BUILD_VERSION=0.12.0
sudo rm -rf torchvision
git clone --branch v$BUILD_VERSION https://github.com/pytorch/vision torchvision
cd torchvision
sudo -H python3 setup.py bdist_wheel
cd dist
sudo pip3 install *.whl
# sudo -H python3 -m pip install torchvision

# Install torch2trt
echo -e "\e[45m Install torch2trt package \e[0m"
cd $HOME/Downloads
sudo rm -rf torch2trt
sudo -H python3 -m pip install packaging
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo -H python3 setup.py install

# Install pycuda
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/
echo -e "\e[45m Install pycuda package \e[0m"
set -e

cd $HOME/Downloads
VER_PYCUDA="v2024.1.2"

if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

echo "** Install requirements"
sudo apt-get install -y build-essential python3-dev
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo pip3 install setuptools

sudo rm -rf pycuda
git clone -b $VER_PYCUDA --recurse-submodules https://github.com/inducer/pycuda.git
cd pycuda
python3 configure.py --cuda-root="/usr/local/cuda-10.2/" 
sudo make install

sudo pip3 install "onnx==1.11.0"