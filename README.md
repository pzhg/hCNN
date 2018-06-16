# hCNN
Hybrid CNN, a MATLAB CNN toolbox that supports complex valued data and insertion of Signal Processing Modules.

## Supported Layers
* Convolutional Layer
* Pooling Layer
* Activation Layer
* Fully Connected Layer
* Reshape Layer
* SoftMax Layer

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
* 'Sigmoid'
	Sigmoid
* 'ABS'
	Absolute value

## Supported Output Layer
* SoftMax

## Insertion of SP Modules
See examples of radar data layer:
>	cnnAddRadarLayer.m  
>	cnnConvolveRadar.m  
>	cnnDeconvolveRadar.m

## Example
See the following file as an example of utilizing this toolbox:
>	'example_MNIST.m'

## Roadmap
* RMSE output layer
* More training algorithms
