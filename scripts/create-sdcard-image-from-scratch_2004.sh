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
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
# sleep 5m

cuterbot_repo=$HOME/cuterbot_repo
cuterbot_home=$HOME/Cuterbot_Demo


sudo mkdir -p $cuterbot_repo

# Install pip and some python dependencies
echo -e "\e[104m Install pip and some python dependencies \e[0m"
sudo apt update
# sudo apt install -y python3-pip python3-pil
# sudo apt install -y python3-pip
# sudo -H python3 -m pip install pip -U
sudo -H python3 -m pip install pillow
# sudo -H python3 -m pip install "Cython<3"
# sudo -H python3 -m pip install --upgrade "numpy<1.19.0"

# Install jtop
echo -e "\e[100m Install jtop \e[0m"
sudo -H python3 -m pip install jetson-stats

cd $HOME
# Install the pre-built TensorFlow pip wheel
echo -e "\e[48;5;202m Install the pre-built TensorFlow pip wheel \e[0m"
sudo pip3 install tensorflow-2.6.5-cp38-cp38-linux_aarch64.whl

# Install the pre-built PyTorch pip wheel 
# download pytorch and torchvision from https://github.com/Qengineering/PyTorch-Jetson-Nano?tab=readme-ov-file
# and guide from https://qengineering.eu/install-pytorch-on-jetson-nano.html
echo -e "\e[45m Install the pre-built PyTorch pip wheel  \e[0m"
sudo pip3 install torch-1.13.0a0+git7c98e70-cp38-cp38-linux_aarch64.whl

# Install torchvision package
echo -e "\e[45m Install torchvision package \e[0m"
sudo pip3 install torchvision-0.14.0a0+5ce4506-cp38-cp38-linux_aarch64.whl


# Install torch2trt
echo -e "\e[45m Install torch2trt package \e[0m"
cd $cuterbot_repo
sudo rm -rf torch2trt
sudo apt-get install libopenblas-dev
sudo -H python3 -m pip install packaging numpy -U
sudo git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo -H python3 setup.py install


# Install traitlets (master, to support the unlink() method)
echo -e "\e[48;5;172m Install traitlets \e[0m"
cd $cuterbot_repo
# sudo python3 -m pip install git+https://github.com/ipython/traitlets@master
# sudo python3 -m pip install git+https://github.com/ipython/traitlets@dead2b8cdde5913572254cf6dc70b5a6065b86f8
sudo rm -rf traitlets
sudo git clone https://github.com/ipython/traitlets.git
cd traitlets
sudo pip3 install -e .


# Install jupyter lab
echo -e "\e[48;5;172m Install Jupyter Lab and extension jupyterlab_widgets \e[0m"
cd $HOME
sudo rm -rf /usr/local/share/jupyter
sudo apt install -y curl
# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -      # use setup_14.x to provide GLIC_28
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
sudo mkdir -p /etc/apt/keyrings
if [ -f /etc/apt/keyrings/nodesource.gpg ]; then 
  sudo rm /etc/apt/keyrings/nodesource.gpg
fi
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
NODE_MAJOR=20
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
sudo apt-get update && sudo apt-get install libffi-dev nodejs -y

sudo -H python3 -m pip install jupyter jupyterlab jupyter_packaging


# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
# sudo -H jupyter labextension install @jupyter-widgets/jupyterlab-manager
sudo -H python3 -m pip install jupyterlab_widgets      # not work properly with jupyterlab blower

echo -e "\e[42m Install jupyter_boken \e[0m"
cd $HOME
# sudo -H python3 -m pip uninstall bokeh
# sudo jupyter labextension install @bokeh/jupyter_bokeh
# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
sudo -H python3 -m pip install jupyter_bokeh
# sudo -H jupyter lab build

jupyter lab --generate-config
python3 -c "from jupyter_server.auth.security import set_password; set_password('$password', '$HOME/.jupyter/jupyter_notebook_config.json')"

# fix for permission error
sudo chown -R cuterbot:cuterbot ~/.local/share/
# sudo chown -R jetbot:jetbot ~/.local/share/

# Install jupyter_clickable_image_widget
echo -e "\e[42m Install jupyter_clickable_image_widget \e[0m"
cd $cuterbot_repo
sudo apt install libssl1.0-dev
sudo rm -rf jupyter_clickable_image_widget
sudo git clone https://github.com/jaybdub/jupyter_clickable_image_widget
cd jupyter_clickable_image_widget
sudo git checkout tags/v0.1
sudo -H python3 -m pip install -e .
sudo -H jupyter labextension install js
sudo -H jupyter lab build

# echo -e "\e[42m Install jupyter_boken \e[0m"
# cd
# sudo -H python3 -m pip install "pillow<9.4.0" jupyter_packaging
# sudo -H python3 -m pip install bokeh
# sudo jupyter labextension install @bokeh/jupyter_bokeh
# sudo -H python3 -m pip install "jupyter_bokeh<2.4.0"
# sudo -H python3 -m pip install jupyter_bokeh
# sudo -H jupyter lab build

# install jetbot python module
echo -e "\e[42m Install jetbot python module \e[0m"
cd $HOME
sudo apt install -y python3-smbus
sudo -H python3 -m pip install pyserial
cd $cuterbot_home
sudo apt install -y cmake
# sudo python3 setup.py install
sudo python3 setup.py bdist_wheel
sudo pip3 install dist/*.whl

# Install jetbot services
echo -e "\e[42m Install jetbot services \e[0m"
cd $cuterbot_home/jetbot/utils
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
cp -r $cuterbot_home/notebooks ~/Notebooks

echo -e "\e[42m All done! \e[0m"

#record the time this script ends
date
