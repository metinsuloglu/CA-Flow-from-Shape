## PointNet++

### Architecture
The PointNet++ model hierarchically groups points together to learn complex features and uses feature propagation layers to
generate predictions on the original point cloud. Unlike the DeepSphere model, no complex pre-processing is necessary, but the point clouds
must be normalised and the number of points in each sample must be equal.

<p align="center">
<img src="https://github.com/metinsuloglu/CA-Flow-from-Shape/blob/master/pointnet_model/images/pointnet_architecture.jpg" alt="model_architecture" width="90%"/>
</p>

### Quick Start
The model uses CUDA and requires a GPU. The code must be executed using Python 2 with TensorFlow 1.x.

To use the model, the custom TensorFlow operations under the folder `tf_ops` must first be compiled.
To do this, set environment variables:

    import os
    import tensorflow as tf
    os.environ["TF_INC"] = tf.sysconfig.get_include()
    os.environ["TF_LIB"] = tf.sysconfig.get_lib()

Then execute the script `tf_xxx_compile.sh` in each subfolder (the location of `cuda-10.1` may need to be altered).

There is currently a pre-trained model in the folder called `log`. If you would like to evaluate this model, create a the folder `/data/ca_data` and add 
a collection of CA point clouds and ground truth values as a number of .txt files (x,y,z values and TAWSS measurements). Make sure the file names are in
the format expected by the code (`case_xx.txt`). Then, execute:

    python train_test.py False
    
The argument specifies whether the model should be trained.
If you would like to train a new model on the same data and then evaluate it, run:

    python train_test.py True
    
Warning: This will overwrite the currently saved model.
After execution the code will save the test set results to a file in the `log` folder called `test_results`.

### Important files

+ wss_pred/train_test.py - *train and then test the PointNet++ model. This file contains the hyperparameters of the model.*
+ log/model.ckpt.xxx - *the PointNet++ model pre-trained on 38 saccular CA point clouds with CFD simulations as ground truth values*
+ models/pointnet2_wss_reg.py - *PointNet++ model definition*
