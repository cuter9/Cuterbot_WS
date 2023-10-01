#!/bin/bash

set -e

password='cuterbot'

# Record the time this script starts
date
# Get the full dir name of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $DIR && sleep 5m

# Keep updating the existing sudo time stamp
sudo -v
while true; do sudo -n true; sleep 120; kill -0 "$$" || exit; done 2>/dev/null &

# Enable i2c permissions
echo -e "\e[100m Enable i2c permissions \e[0m"
sudo usermod -aG i2c $USER

# Make swapfile
cd
sudo swapoff -a
sudo fallocate -l 10G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
# sleep 5m

# Install pip and some python dependencies
echo -e "\e[104m Install pip and some python dependencies \e[0m"
sudo apt update
# sudo apt install -y python3-pip python3-pil
sudo -H python3 -m pip install pillow
sudo -H python3 -m pip install Cython
sudo -H python3 -m pip install --upgrade numpy

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

# Install traitlets (master, to support the unlink() method)
echo -e "\e[48;5;172m Install traitlets \e[0m"
#sudo python3 -m pip install git+https://github.com/ipython/traitlets@master
sudo python3 -m pip install git+https://github.com/ipython/traitlets@dead2b8cdde5913572254cf6dc70b5a6065b86f8

# Install jupyter lab
echo -e "\e[48;5;172m Install Jupyter Lab and extension jupyterlab_widgets \e[0m"
sudo rm -rf /usr/local/share/jupyter
sudo apt install -y curl
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -      # use setup_14.x to provide GLIC_28
sudo apt install -y nodejs libffi-dev
sudo -H python3 -m pip install jupyter jupyterlab jupyter_packaging

# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
sudo -H jupyter labextension install @jupyter-widgets/jupyterlab-manager
# sudo -H python3 -m pip install jupyterlab_widgets      # not work properly with jupyterlab blower

# sudo -H python3 -m pip uninstall bokeh
# sudo jupyter labextension install @bokeh/jupyter_bokeh
# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
# sudo -H jupyter lab build

jupyter lab --generate-config
python3 -c "from notebook.auth.security import set_password; set_password('$password', '$HOME/.jupyter/jupyter_notebook_config.json')"

# fix for permission error
sudo chown -R cuterbot:cuterbot ~/.local/share/
# sudo chown -R jetbot:jetbot ~/.local/share/

# Install jupyter_clickable_image_widget
echo -e "\e[42m Install jupyter_clickable_image_widget \e[0m"
cd
sudo apt install libssl1.0-dev
sudo rm -rf jupyter_clickable_image_widget
git clone https://github.com/jaybdub/jupyter_clickable_image_widget
cd jupyter_clickable_image_widget
git checkout tags/v0.1
sudo -H python3 -m pip install -e .
sudo -H jupyter labextension install js
sudo -H jupyter lab build

echo -e "\e[42m Install jupyter_boken \e[0m"
cd
# sudo -H python3 -m pip install "pillow<9.4.0" jupyter_packaging
# sudo -H python3 -m pip install bokeh
# sudo jupyter labextension install @bokeh/jupyter_bokeh
# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
sudo -H python3 -m pip install jupyter_bokeh
sudo -H jupyter lab build

# install jetbot python module
echo -e "\e[42m Install jetbot python module \e[0m"
cd
sudo apt install -y python3-smbus
sudo -H python3 -m pip install pyserial
cd ~/Cuterbot
sudo apt install -y cmake
sudo python3 setup.py install

# Install jetbot services
echo -e "\e[42m Install jetbot services \e[0m"
cd ~/Cuterbot/jetbot/utils
python3 create_stats_service.py
sudo mv jetbot_stats.service /etc/systemd/system/jetbot_stats.service
sudo systemctl enable jetbot_stats
sudo systemctl start jetbot_stats
python3 create_jupyter_service.py
sudo mv jetbot_jupyter.service /etc/systemd/system/jetbot_jupyter.service
sudo systemctl enable jetbot_jupyter
sudo systemctl start jetbot_jupyter


# install python gst dependencies
sudo apt install -y \
    libwayland-egl1 \
    gstreamer1.0-plugins-bad \
    libgstreamer-plugins-bad1.0-0 \
    gstreamer1.0-plugins-good \
    python3-gst-1.0
    
# install zmq dependency (should actually already be resolved by jupyter)
sudo -H python3 -m pip install pyzmq
    

# Optimize the system configuration to create more headroom
sudo nvpmodel -m 0
# sudo systemctl set-default multi-user
sudo systemctl set-default graphical.target    # enable GUI
sudo systemctl disable nvzramconfig.service

# Copy JetBot notebooks to home directory
sudo rm -rf ~/Notebooks
cp -r ~/Cuterbot/notebooks ~/Notebooks

echo -e "\e[42m All done! \e[0m"

#record the time this script ends
date
