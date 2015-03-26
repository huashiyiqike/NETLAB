# NETLAB

A C++ neural network libray for sequences mainly, 
focusing on Long Short-term Memory (LSTM)
and Restricted Boltsmann Machine (RBM) Layers.  

Creater&Maintainer: Qi Lyu

Features
=====
* Easy to create new layers with Matlab like syntax (just need to preallocate memory)
* Automatic Backpropagation through time (BPTT) calculation between layers
* Unified CPU/GPU code: could run in both CPU and GPU
* Extendable: easy to configure the net parameter with google's protobuf  
* Weight snapshot, easy for loading model and visualizing 

Dependencies
====
* CUDA library version 6.5 (recommended), 6.0, 5.5, or 5.0 and the latest driver version for CUDA 6 or 319.* for CUDA 5 (and NOT 331.*)
* BLAS (provided via ATLAS, MKL, or OpenBLAS)
* Boost (>= 1.55)
* mshadow (https://github.com/tqchen/mshadow)
* google-glog, Protocol Buffers, LevelDB (not necessary)
* OpenMp

Related Projects
=====
* CXXNET: neural network implementation based on mshadow: https://github.com/antinucleon/cxxnet (Some of the design
 come from here, and a few lines are borrowed from this project)
 
Disclaimer
====
This code is only for research purpose; 
It may not be the version I am using and I probably have no time to maintain it anymore. 
Please note this code should be used at your own risk. 
