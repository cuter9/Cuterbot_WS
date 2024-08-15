# Make swapfile
cd
sudo swapoff -a
sudo fallocate -l 6G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
# sleep 5m

# Install pip and some python dependencies
echo -e "\e[104m Install pip and some python dependencies \e[0m"
sudo apt update
# sudo apt install -y python3-pip python3-pil
sudo apt install -y python3-pip
sudo -H python3 -m pip install pip -U
sudo -H python3 -m pip install pillow
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
echo -e "\e[45m Install the pre-built PyTorch pip wheel  \e[0m"
cd
# wget -N https://nvidia.box.com/shared/static/yr6sjswn25z7oankw8zy1roow9cy5ur1.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl
wget -N https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install -y libopenblas-base libopenmpi-dev 
# sudo -H python3 -m pip install Cython
sudo -H python3 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl 

# Install torchvision package
echo -e "\e[45m Install torchvision package \e[0m"
cd
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo rm -rf torchvision
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
# git clone https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
#git checkout v0.4.0
sudo -H python3 setup.py install
# sudo -H python3 -m pip install torchvision

# Install torch2trt
echo -e "\e[45m Install torch2trt package \e[0m"
cd $HOME
sudo rm -rf torch2trt
sudo -H python3 -m pip install packaging
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo -H python3 setup.py install

# Install pycuda
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/
echo -e "\e[45m Install pycuda package \e[0m"
set -e

if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

echo "** Install requirements"
sudo apt-get install -y build-essential python3-dev
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo pip3 install setuptools

wget https://files.pythonhosted.org/packages/3f/50/dd356c8afe228baecaf0259b9579121dd869c5ace07a296158c39ac5065a/pycuda-2024.1.tar.gz
tar xvf pycuda-2024.1.tar.gz

cd pycuda-2024.1
python3 configure.py --cuda-root="/usr/local/cuda-10.2/" 
sudo make install