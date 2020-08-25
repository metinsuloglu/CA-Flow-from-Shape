## DeepSphere U-Net

### Idea
The cerebral aneurysm (CA) point clouds do not have an equal number of points and have no connectivity information. If we can find a point to point correspondence
between the CA shapes and a template object with a fixed number of points, we can transform the CAs and apply deep learning models on the template shape.
In this project we encode the anerysm morphology on a sphere.

<p align="center">
<img src="https://github.com/metinsuloglu/CA-Flow-from-Shape/blob/master/deepsphere_model/images/correspondences.jpg" alt="point_correspondences" width="30%"/>
</p>

The code in this folder implements the steps for mapping a CA point cloud onto a sphere and training the DeepSphere model using spherical graph convolutions.

### Quick Start
Part of the code in this folder is taken from the [DeepSphere model repository](https://github.com/deepsphere/deepsphere-pytorch). Therefore
the requirements are the same:

- Python 3
- PyTorch >= 1.3.1
- PyGSP fork: https://github.com/Droxef/pygsp/tree/new_sphere_graph
- torchvision >= 0.4.2
- cudatoolkit >= 10.0 (for running on a GPU)

Also, the algorithms use NumPy / SciPy. For visualising results make sure either Open3D or Matplotlib is installed. 

Create a folder `/deepsphere/data/ca_data/` and add the CA shape data as a collection of .txt files, where the rows store 4 values seperated by whitespace:
(x,y,z) coordinates and a TAWSS value.

You can change the hyperparameters of the model in `config.example.yml`. To train and validate the model, in this directory run:

    python train.py --config-file config.example.yml

and to test the model:

    python test.py --config-file config.example.yml

The default behaviour of the test script is to use the pre-trained model named `pretrained_model_38_cases.pt`.

### Important files

+ train.py - *train the U-Net DeepSphere model*
+ test.py - *load and evaluate the model*
+ pretrained_model_38_cases.pt - *a model pre-trained on 38 saccular CA point clouds*
+ deepsphere/data/dataset.py - *script used to load the data files*
+ deepsphere/data/process.py - *contains the pre-processing steps and the algorithm for finding point correspondences*
+ deepsphere/models/spherical_unet/unet_model.py  - *U-Net model definition*

<p align="center">
<img src="https://github.com/metinsuloglu/CA-Flow-from-Shape/blob/master/deepsphere_model/images/deepsphere_architecture.jpg" alt="model_architecture" width="80%"/>
</p>
