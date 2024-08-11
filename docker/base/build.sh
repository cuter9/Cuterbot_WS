cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc ../.. # copy to jetbot root
# sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
# sudo apt-get install kitware-archive-keyring
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update

docker build \
    --build-arg BASE_IMAGE="$JETBOT_BASE_IMAGE" \
    -t "$JETBOT_DOCKER_REMOTE"/jetbot:base-"$JETBOT_VERSION"-$L4T_VERSION \
    -f Dockerfile \
    ../..  # jetbot repo root as context

