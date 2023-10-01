# ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ====================
# INSTALL COMMON TOOLS
# ====================
RUN apt-get install software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

RUN apt-get update && apt-get install -y net-tools nano cmake


# ==================
# INSTALL TENSORFLOW
# ==================
ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"
ARG MAKEFLAGS=-j6
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  gfortran \
		  build-essential \
		  liblapack-dev \ 
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  zip \
		  libjpeg8-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN pip3 install setuptools Cython wheel
RUN python3 -m pip install pip -U   # upgrade pip for following installation of protobuf
RUN pip3 install setuptools 'Cython==0.29.35' wheel
RUN pip3 install 'numpy<1.19.0'
RUN pip3 install h5py==2.10.0
# RUN pip3 install future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.3.3 futures protobuf pybind11
RUN pip3 install future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.3.3 futures==2.2.0 protobuf pybind11
# RUN pip3 install numpy

# ARG TENSORFLOW_URL=https://nvidia.box.com/shared/static/rummpy6q1km1wivomalpkwt2jy28mndf.whl 
# ARG TENSORFLOW_URL= https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl
# ARG TENSORFLOW_WHL=tensorflow-1.15.2+nv-cp36-cp36m-linux_aarch64.whl

RUN python3 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 'tensorflow<2'
# RUN wget --quiet --show-progress ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL} && \
 #   pip3 install ${TENSORFLOW_WHL} && \
 #   rm ${TENSORFLOW_WHL}


# ==============
# INSTALL OPENCV
# ==============
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*


# =================
# INSTALL GSTREAMER
# =================
RUN apt-get update && \
    apt-get install -y \
    libwayland-egl1 \
    gstreamer1.0-plugins-bad \
    libgstreamer-plugins-bad1.0-0 \
    gstreamer1.0-plugins-good \
    python3-gst-1.0


# =======================
# INSTALL JUPYTER RELATED
# =======================

# Install traitlets (master, to support the unlink() method)
RUN echo -e "\e[48;5;172m Install traitlets \e[0m"
#sudo python3 -m pip install git+https://github.com/ipython/traitlets@master
RUN python3 -m pip install git+https://github.com/ipython/traitlets@dead2b8cdde5913572254cf6dc70b5a6065b86f8

# Install jupyter lab
RUN echo -e "\e[48;5;172m Install Jupyter Lab \e[0m"
RUN apt-get update
RUN apt-get install -y curl
# RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs libffi-dev
# RUN pip3 install jupyter jupyterlab==2.2.6
RUN pip3 install jupyter jupyterlab jupyter_packaging
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN pip3 install ipywidgets -U      # update the latest ipython widgets



# Install jupyter_clickable_image_widget
RUN echo "\e[42m Install jupyter_clickable_image_widget \e[0m"
RUN cd && \
    apt-get install -y libssl1.0-dev && \
    git clone https://github.com/jaybdub/jupyter_clickable_image_widget && \
    cd jupyter_clickable_image_widget && \
    git checkout tags/v0.1 && \
    pip3 install -e . && \
    jupyter labextension install js && \
    jupyter lab build
    
# ============================
# INSTALL MISC DEPENDENCIES
# ============================
RUN apt-get update && apt-get install -y supervisor unzip
RUN apt install -y python3-smbus && pip3 install pyzmq


# =================
# INSTALL TORCH2TRT
# =================
ENV TORCH2TRT_REPO_DIR=/opt/
RUN cd ${TORCH2TRT_REPO_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install


# ==============
# INSTALL JETBOT
# ==============
RUN python3 -m pip install pyserial
RUN python3 -m pip install matplotlib
ENV JETBOT_REPO_DIR=/opt/jetbot
COPY . ${JETBOT_REPO_DIR}
RUN cd ${JETBOT_REPO_DIR} && python3 setup.py install

# COPY ../notebooks Notebooks/


