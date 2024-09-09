cd $HOME/Downloads
# ENV QWIIC_I2C_DIR=/opt/Qwiic_Py/qwiic_i2c
if [ ! -d "./Qwiic_Py" ]; then 
  git clone --recurse-submodules https://github.com/sparkfun/Qwiic_Py.git 
fi
cd ./Qwiic_Py

pushd ./qwiic/drivers
if [ ! -d "./qwiic_as6212" ]; then 
  rm -rf ./qwiic_as6212
fi
if [ ! -d "./qwiic_kx13x" ]; then
  rm -rf ./qwiic_kx13x
fi

popd
python3 setup.py bdist_wheel && cd dist && sudo pip3 install *.whl && cd ..

pushd ./qwiic_i2c
python3 setup.py bdist_wheel && cd dist && sudo pip3 install *.whl

# cd $HOME/Downloads && rm -rf Qwiic_Py
sudo pip3 install Adafruit_MotorHat
sudo pip3 install Adafruit-SSD1306
