# ARG BASE_IMAGE=jetbot-base:jp44
ARG BASE_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

CMD cd ${JETBOT_REPO_DIR} && python3 scripts/zmq_camera_publisher.py
