#! /bin/bash

# shellcheck disable=SC2164
cd ~
rm -r /opt/jetbot
cp -r /workspace/Cuterbot_2004 /opt/jetbot
# shellcheck disable=SC2164
cd /opt/jetbot
python3 setup.py bdist_wheel
cd dist
pip3 install jetbot-0.4.3-py3-none-any.whl --force-reinstall

