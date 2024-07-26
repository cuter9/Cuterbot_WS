sudo pip3 install jetson-stats -U
cd base && ./build.sh && cd ..
cd models && ./build.sh && cd ..
cd display && ./build.sh && cd ..
cd jupyter && ./build.sh && cd ..
cd camera && ./build.sh && cd ..
