# hCNN, Hybrid Neural Network Toolbox

hCNN, Hybrid Neural Network, a MATLAB NN toolbox that supports complex valued data and insertion of Signal Processing Modules.

**GPU supported. Please use a CUDA enabled device and `to.useGPU=1` if you want to enable it.**

## Supported Layers
* Convolutional Layer

`comp='c'` if you want the parameters to be complex.

```
cnn = cnnAddConvLayer(cnn, [3, 3], 8, 'r');
% Add a convolution layer with 3*3 filter and 8 channels, real ('c' for complex).
```

* Pooling Layer

* Activation Layer

* Fully Connected Layer

`comp='c'` if you want the parameters to be complex.

```
cnn = cnnAddFCLayer(cnn, 128, 'r');
% Add a fully connected layer with 128 outputs, real ('c' for complex).
```

* Reshape Layer

* Output Layer

* Batched Normalization Layer

* Drop Out Layer

* Multiple Channel (BLOB) Layer

* Signal Processing Modules: Transformation Layer

## Supported Pooling Methods
* 'mean'

	Mean pooling

* 'max'

	Max pooling

## Supported Activation Functions
* 'ReLu'

	ReLu

* 'LReLu'

	Leaked ReLu

	The leaky rate can be adjusted in `cnnAddActivationLayer.m`

* 'Sigmoid'

	Sigmoid

* 'ABS'

	Absolute value

You can add more activation functions in
>	`cnnAddActionvationLayer.m`  
>	`cnnActivate.m`  
>	`cnnDeActivate.m` 

## Supported Output Layer
* SoftMax (Cross Entropy)
* MSE

## Multiple Channel (BLOB) Layer

The layer is constructed of multiple sub-NNs as multiple channels as:

```
           /--- sub NN ---\
          /                \
   Input------- sub NN ---------Output
          \                /
	       \--- sub NN ---/
```

Each sub-NN can also contain of such BLOB Layer, resulting a nested structure. See `example_MicroDoppler.m` for detail.

## Signal Processing (SP) Modules

### General SP Layers: contain parameters that can be trained during the training of hybrid-NN.

See the example of Radar Data Layer:
>	`cnnAddRadarLayer.m`  
>	`cnnConvolveRadar.m`  
>	`cnnDeconvolveRadar.m`  

You can also add your own SP layers. Use the above examples as references.

### Special SP Layers: does not contain trainable parameters

* Compresse Sensing (CS) Layer
* Covariance-PCA Layer
* Transformation Layer
	* PCA Layer
	* FFT Layer
	* DWT Layer
	* ABS/ARG/REAL/IMAG Layer
	* LOWPASS/HIPASS Layer

	See `cnnAddTransformLayer.m` for detail. 
	
More special SP layers and transformation types can be added manually. 

See examples of Special SP layer:
>	`cnnAddCSLayer.m` `cnnCS.m`  
>	`cnnAddCoPCALayer.m` `cnnCoPCA.m`  
>	`cnnAddTransformLayer.m` `cnnTransform.m`

## Supported Training Method
* BP (Gradient descent)
 
## Example
See the following file as an example of utilizing this toolbox:
>	`example_MNIST.m` 

Dataset used in this example is from [here](http://yann.lecun.com/exdb/mnist/).

See the following file as an example of utilization of Multiple channel (BLOB) layer and the insertion of SP Module:
>	`example_MicroDoppler.m`

Dataset used in this example is from [here](https://www.mathworks.com/help/phased/examples/pedestrian-and-bicyclist-classification-using-deep-learning.html?s_eid=PEP_16543).

You can download the MAT file used in this example from [here](https://1drv.ms/u/s!Akr-loZjbPYVufFK8J6pAMtHi1fEyA?e=T6Vgss).

## Roadmap
* More training algorithms

## Reference

- [1] Zhang, Z., Jian, M., Lu, Z., Chen, H., James, S., Wang, C., & Gentile, R. (2020, April). Embedded Micro Radar for Pedestrian Detection in Clutter. In 2020 IEEE International Radar Conference (RADAR) (pp. 368-372). IEEE.

- [2] Zhang, Z., Chen, X., & Tian, Z. (2018, November). A hybrid neural network framework and application to radar automatic target recognition. In 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP) (pp. 246-250). IEEE.
