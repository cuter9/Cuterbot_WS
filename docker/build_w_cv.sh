#!/bin/bash
cd base && ./build_w_cv.sh && cd ..
cd models && ./build.sh && cd ..
cd display && ./build.sh && cd ..
cd jupyter && ./build.sh && cd ..
cd camera && ./build.sh && cd ..
