#/bin/bash
/usr/local/cuda-10.1/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /tensorflow-1.15.2/python2.7/tensorflow_core/include -I /usr/local/cuda-10.1/include -I$TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
