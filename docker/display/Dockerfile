# ARG BASE_IMAGE=jetbot-base:jp44
ARG BASE_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}
RUN python3 -m pip install jetson-stats -U

CMD python3 -m jetbot.apps.stats
