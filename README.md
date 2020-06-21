# hCNN, Hybrid Neural Network Toolbox

hCNN, Hybrid Neural Network, a MATLAB NN toolbox that supports complex valued data and insertion of Signal Processing Modules.

**GPU supported. Please use a CUDA enabled device and `to.useGPU=1` if you want to enable it.**

## Supported Layers
* Convolutional Layer

`comp='c'` if you want the parameters to be complex.

```
cnn = cnnAddConvLayer(cnn, [3, 3], 8, 'r');
% Add a convolution layer with 3\times 3 filter and 8 channels, real ('c' for complex).
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

* SoftMax Layer

* Batched Normalization Layer

* Drop Out Layer

* Multiple Channel (BLOB) Layer like this:

```
         /--- sub NN ---\
        /                \
   NN-------- sub NN --------(Other structutres)--Output
        \                 /
	     \--- sub NN ----/
```

Each sub-NN can also be composed of such multiple channel (BLOB) Layer.

* Transformation Layer
	* CS Layer
	* Covariance-PCA Layer
	* FFT Layer
	* PCA/FFT/DWT/ABS/ARG/REAL/IMAG/LOWPASS/HIPASS... Layer

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
* SoftMax

## Supported Training Method
* BP (Gradient descent)

## Insertion of SP Modules
See examples of radar data layer:
>	`cnnAddRadarLayer.m` 
>	`cnnConvolveRadar.m`  
>	`cnnDeconvolveRadar.m` 

See examples of Transformation layer:
>	`cnnAddTransformLayer.m` 

>	`cnnTransform.m`

## Example
See the following file as an example of utilizing this toolbox:
>	`example_MNIST.m` 

## Roadmap
* RMSE output layer
* More training algorithms

## Reference

- [1] Zhang, Z., Jian, M., Lu, Z., Chen, H., James, S., Wang, C., & Gentile, R. (2020, April). Embedded Micro Radar for Pedestrian Detection in Clutter. In 2020 IEEE International Radar Conference (RADAR) (pp. 368-372). IEEE.

- [2] Zhang, Z., Chen, X., & Tian, Z. (2018, November). A hybrid neural network framework and application to radar automatic target recognition. In 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP) (pp. 246-250). IEEE.