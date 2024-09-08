cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc ../.. # copy to jetbot root
# sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
# sudo apt-get install kitware-archive-keyring

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update
sudo apt-get update
sudo apt-get install python3-pip -y
sudo pip3 install pip -U

export HOME=/home/cuterbot

# do below before install latest opencv v3.9.0:
# download Video_Codec_SDK and copy lib(.so) and .h file to cuda_10.2
# create symlink to the Video_Codec_SDK lib
# after above the Video_Codec_SDK lib, the nvidia container toolkit will automatically map into container when building cuterbot
# install the requirements for python-opencv
if [ ! -d "${HOME}/Cuterbot_Demo/repo" ]; then
  gdown --no-cookies --folder https://drive.google.com/drive/folders/1d1KtLKNBpOoTWb-Z1_pAaMtCIzfE6LyC -O ${HOME}/Cuterbot_Demo/repo
  pushd ${HOME}/Cuterbot_Demo/repo && unzip Video_Codec_SDK_12.1.14.zip && mv Video_Codec_SDK_12.1.14 Video_Codec_SDK
  pushd ./Video_Codec_SDK
  sudo cp ./Interface/*.h /usr/local/cuda-10.2/targets/aarch64-linux/include/
  sudo cp ./Lib/linux/stubs/aarch64/*.so /usr/local/cuda-10.2/targets/aarch64-linux/lib/
  
  if [ -f "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvcuvid.so.1" ]; then
    sudo rm "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvcuvid.so.1"
  fi
  sudo ln -s /usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvcuvid.so /usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvcuvid.so.1
  
  
  if [ -f "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvidia-encode.so.1" ]; then
    sudo rm "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvidia-encode.so.1"
  fi
  sudo ln -s /usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvidia-encode.so /usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvidia-encode.so.1
  
  popd
  popd
fi

docker build \
    --build-arg BASE_IMAGE="$JETBOT_BASE_IMAGE" \
    -t "$JETBOT_DOCKER_REMOTE"/jetbot:base-"$JETBOT_VERSION"-$L4T_VERSION \
    -f Dockerfile_w_cv \
    ../.. 2>&1 | tee build_cuterbot_docker.log   # jetbot repo root as context