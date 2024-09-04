cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc ../.. # copy to jetbot root
# sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
# sudo apt-get install kitware-archive-keyring

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update
export HOME=/home/cuterbot

gdown --no-cookies --folder https://drive.google.com/drive/folders/1d1KtLKNBpOoTWb-Z1_pAaMtCIzfE6LyC -O ${HOME}/repo
pushd ${HOME}/repo && unzip Video_Codec_SDK_12.1.14.zip && mv Video_Codec_SDK_12.1.14 Video_Codec_SDK
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
docker build \
    --build-arg BASE_IMAGE="$JETBOT_BASE_IMAGE" \
    --build-arg HOME=$HOME \
    -t "$JETBOT_DOCKER_REMOTE"/jetbot:base-"$JETBOT_VERSION"-$L4T_VERSION \
    -f Dockerfile_w_cv \
    $HOME 2>&1 | tee build_cuterbot_docker.log   # jetbot repo root as context

