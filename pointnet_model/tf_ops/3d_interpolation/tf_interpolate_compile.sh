g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /tensorflow-1.15.2/python2.7/tensorflow_core/include -I /usr/local/cuda-10.1/include -I$TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
