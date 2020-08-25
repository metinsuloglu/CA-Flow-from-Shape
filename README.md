# Geometric deep learning on cerebral aneurysm point clouds
An investigation into applying deep learning to predict wall shear stress over cerebral aneurysms.

Part of my Master's Thesis at the University of Leeds.

## Introduction
The aim of the project is to examine ways of predicting time-averaged wall shear stress given the morphology of a cerebral aneurysm (CA).
This repository contains two folders for different deep learning architectures:

- [PointNet++](https://github.com/charlesq34/pointnet2)
- [DeepSphere U-Net](https://github.com/deepsphere/deepsphere-pytorch)

The PointNet++ model expects fixed-sized point clouds as input and outputs the predicted TAWSS values at each point. 
DeepSphere first projects the CA onto a sphere (a subdivided icosahedron) and passes the transformed representation through 
spherical convolution and pooling operations.

Each folder contains the code for training and testing the models. Also, the folders contain saved models which have been pre-trained on 38 saccular CA
shapes taken from a cohort of patients, with computational fluid dynamics (CFD) simulation results assumed to be the ground truth values.

Please see the README files in each folder for more information on running the models and the requirements.