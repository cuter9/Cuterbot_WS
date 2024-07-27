sudo docker run -it -d \
    --restart always \
    --runtime nvidia \
    --network host \
    --privileged \
    --name=jetbot_display \
    --env DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME"/.Xauthority:/root/.Xauthority \
    -v /run/jtop.sock:/run/jtop.sock \
    "$JETBOT_DOCKER_REMOTE"/jetbot:display-"$JETBOT_VERSION"-"$L4T_VERSION"

sudo systemctl restart jtop.service