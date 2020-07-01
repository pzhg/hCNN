# hCNN, Hybrid Neural Network Toolbox

hCNN, Hybrid Neural Network, a MATLAB NN toolbox that supports complex valued data and insertion of Signal Processing Modules.

**GPU supported. Please use a CUDA enabled device and `to.useGPU=1` if you want to enable it.**

## Overview

During the recent decade, deep learning technology, particularly deep neural network (DNN), has gained tremendous popularity in various fields including signal processing (SP). As a data-driven framework, DNN treats the learning problem as a “black-box” that extracts useful features directly from data. With sufficient training, DNN does not rely on any special structure or property of the processed data, making it universally applicable to diverse problem models. As such, DNN can help to expand the functionality of SP to handle problems that cannot be well-modeled.

As a data-driven approach, DNN suffers from some drawbacks. Usually DNN requires huge amount of data to work effectively, and its training stage is also computation costly. Specializing to the (radar) signal processing field, there are abundant highly-structured or man-made signals with known properties, such as low-rankness or sparsity. DNN, is obviously not as efficient in processing and extracting useful information from those signals with known models and properties. It means that, if we can leverage these known properties, the DNN performance is expected to be improved significantly in terms of required data amount and computation load.

Here we proposed a hybrid neural network (Hybrid-NN) as a novel scheme to improve the detection performance in terms of validation accuracy and required training data amount. The idea is to insert signal processing (SP) modules into conventional neural networks, especially CNNs. The goal is to increase the training/validation accuracy and reduce the required amount of data. Hybrid-NN also supports multiple channel structures, that each channel is composed of sub neural networks (sub-NNs), for specific objectives.

An example of overall architecture of a hybrid-NN is shown in as follows:

<img src="https://pzhg.github.io/hCNN/hybrid-nn-example.png" width=55% align=center alt="Architecture of an example hybrid-NN">
 
This example hybrid-NN has three parallel subnetworks as three channels. The `sub-NN 1` and `sub-NN 2` has a PCA layer at the very beginning as the SP module for some specific objectives, and the `sub-NN 3` is constructed as a traditional CNN which is intended to detect all general targets. The three subnetworks are combined with two fully connected layers.

## Usage

To usage of this toolbox has for steps, which is simple and intuitive:

1. Define hyper parameters and initialize the NN

	```
	to.epochs = 3;              % Epoch number
	to.batch = 400;             % Batch number
	to.batch_size = 150;        % Batch size
	to.alpha = 0.1;             % Learning rate
	to.momentum = 0.9;          % Momentum
	to.mom = 0.5;               % Initial momentum
	to.momIncrease = 20;        % Momemtum change iteration count
	to.lambda = 0.0001;         % Weight decay parameter (a.k.a. L2 regularization parameter)
	to.useGPU = 1;              % Use GPU

	cnn = cnnInit(to);
	```

2. Define NN structure

	```
	cnn = cnnAddInputLayer(cnn, [28, 28], 1);
	cnn = cnnAddConvLayer(cnn, [3, 3], 8, 'r');
	cnn = cnnAddBNLayer(cnn);
	cnn = cnnAddActivationLayer(cnn, 'relu');
	cnn = cnnAddPoolLayer(cnn, 'max', [2, 2]);
	cnn = cnnAddReshapeLayer(cnn);
	cnn = cnnAddFCLayer(cnn, 128, 'r');
	cnn = cnnAddDropOutLayer(cnn, 0.3);
	cnn = cnnAddBNLayer(cnn);
	cnn = cnnAddOutputLayer(cnn, 'softmax');
	```

3. Train

	```
	cnn = cnnInitVelocity(cnn); % Initial the NN Parameters
	[ERR, cnn] = cnnTrainBP(cnn, TrainData, LabelData);
	```

4. Validate

	```
	cnn.to.test = 1;            % Set the test flag to 1
	acc = cnnTestData(cnn, VData, VLabel, 1000);
	```

## Supported Layers
* Convolutional Layer

	```
	cnn = cnnAddConvLayer(NN_NAME, FILTER_SIZE, FILTER_NUM, COMPLEX_FLAG);
	```

	Set `COMPLEX_FLAG` to `'c'` if you want the parameters to be complex, otherwise use `'r'`.

	Example:
	```
	cnn = cnnAddConvLayer(cnn, [3, 3], 8, 'r');
	% Add a convolution layer with 3*3 filter and 8 channels, real.
	```

* Pooling Layer

	```
	cnn = cnnAddPoolLayer(NN_NAME, POOLING_METHOD, POOLING_SIZE);
	```

	See [Supported Pooling Methods Section](### Supported Pooling Methods) for a list of supported pooling methods. 
	Example:
	```
	cnn = cnnAddPoolLayer(cnn, 'max', [2, 2]);
	% Add a pooling layer with max pooling and 2*2 pooling size.
	```

* Activation Layer

	```
	cnn = cnnAddActivationLayer(NN_NAME, ACTIVITAION_FUNCTION);
	```

	See [Supported Activation Functions Section](### Supported Activation Functions) for a list of supported pooling methods. 	
	Example:
	```
	cnn = cnnAddActivationLayer(cnn, 'relu');
	% Add an activation layer with 'reLu' activation function.
	```

* Fully Connected Layer

	```
	cnn = cnnAddFCLayer(NN_NAME, OUTPUT_SIZE, COMPLEX_FLAG);
	```

	Set `COMPLEX_FLAG` to `'c'` if you want the parameters to be complex, otherwise use `'r'`.

	Example:
	```
	cnn = cnnAddFCLayer(cnn, 128, 'r');
	% Add a fully connected layer with 128 outputs, real parameters.
	```

* Reshape Layer

	```
	cnn = cnnAddReshapeLayer(NN_NAME);
	```

	Reshape the 2-D layers (e.g. convolutional layers) to the 1-D fully connected layer for output.

* Output Layer

	```
	cnn = cnnAddOutputLayer(NN_NAME, OUTPUT_METHOD);
	```

	See [Supported Output Methods Section](### Supported Output Methods) for a list of supported output layer types. 	
	Example:
	```
	cnn = cnnAddOutputLayer(cnn, 'softmax');
	% Add a SoftMax output layer.
	```

* Batched Normalization Layer

	```
	cnn = cnnAddBNLayer(NN_NAME);
	cnn = cnnAddBNLayer(NN_NAME, MODE)
	```

	If can manually define `MODE` as `1` for 2-D layers (e.g. convolutional layers) and `2` for 1-D layers (e.g. fully connected layers). If not specified, it is automatically decided by its preceding layer.

* Drop Out Layer

	```
	cnn = cnnAddDropOutLayer(NN_NAME, DROPOUT_RATE);
	```

	`DROPOUT_RATE` specifies the probability of removing neurons.

	Example:
	```
	cnn = cnnAddDropOutLayer(cnn, 0.3);
	% Remove 30% neurons (i.e. keep 70% neurons).
	```

* Multiple Channel (BLOB) Layer

	We use this layer to support the multiple channel structure. See [Multiple Channel (BLOB) Layer Section](## Multiple Channel (BLOB) Layer) detail.

* Signal Processing Layers and Transform Layer

	We use these layers to support the signal processing (SP) modules. See [Signal Processing (SP) Modules Section](## Signal Processing (SP) Modules) detail.

## Layer Options

### Supported Pooling Methods
* `'mean'`

	Mean pooling.

* `'max'`

	Max pooling.

### Supported Activation Functions
* `'relu'`

	ReLu activation.

* `'lrelu'`

	Leaked ReLu activation.

	The leaky rate can be adjusted in `cnnAddActivationLayer.m`

* `'sigmoid'`

	Sigmoid activation.

* `'abs'`

	Absolute value activation.

If your desired activation function is not listed above, you can easily add more activation functions in
>	`cnnAddActionvationLayer.m`  
>	`cnnActivate.m`  
>	`cnnDeActivate.m` 

### Supported Output Methods
* `'softmax'`
	
	SoftMax (Cross Entropy) output.
	
* `'mse'`

	MSE output.

* `'end_BLOB'`

	End layer of sub-NN in a Multiple Channel (BLOB) layer.

## Multiple Channel (BLOB) Layer

The layer is constructed of multiple sub-NNs as multiple channels as:

```
           /--- sub NN ---\
          /                \
   Input------- sub NN ---------Output
          \                /
	       \--- sub NN ---/
```

Each sub-NN can also contain of such BLOB Layer, resulting a nested structure. 

Syntax:

	```
	SUBNET_LIST = {SUBNN_1, SUBNN_2, ...}
	cnn = cnnAddBLOBLayer(cnn, SUBNET_LIST, OUTPUT_DIM, COMBINE_TYPE);
	```

	* `SUBNET_LIST` is a cell of multiple sub-NNs. Eash sub-NN is a conventional NN defined as usual. You must use `end_BLOB` as its output layer type.
	* `OUTPUT_DIM` is the dimention (number of neurons) of the output of this BLOB layer.
	* `COMBINE_TYPE` specifies that how to combine the sub-NNs. See [Supported Combining Types](###Supported Combining Types) for a list of supported combining types.

Example:
	```
	nets = {cnn1, cnn2};
	cnn = cnnAddBLOBLayer(cnn, nets, 128, 2);
	% The BLOB layer has two sub-NNs, the output dimension is 128, and the combining type is 'linking'.
	```

See `example_MicroDoppler.m` for detail usages.

### Supported Combining Types

* `COMBINE_TYPE = 1` 

	The combining type is `'adding'`, which adds the outputs of each sub-NN together and output their summation to the succeeding later. 
	
	In this case, the `OUTPUT_DIM` should be the same as output dimension of each sub-NN.

* `COMBINE_TYPE = 2` 

	The combining type is `'linking'`, which links the outputs of each sub-NN into a long vector. 
	
	In this case, the `OUTPUT_DIM` should be the same as the summation of output dimensions of sub-NNs.

## Signal Processing (SP) Modules

### General SP Layers: contain parameters that can be trained during the training of hybrid-NN.

See the example of Radar Data Layer:
>	`cnnAddRadarLayer.m`  
>	`cnnConvolveRadar.m`  
>	`cnnDeconvolveRadar.m`  

You can also add your own SP layers. Use the above examples as references.

### Special SP Layers: does not contain trainable parameters

Currently the following special SP layers are supported:

* Compressed Sensing (CS) Layer

	```
	cnn = cnnAddCSLayer(NN_NAME, OUTDIM, COMPLEX_FLAG);
	```

	Add a compressed sensing (CS) layer with output dimension `OUTDIM`, and set `COMPLEX_FLAG` to `'c'` if you want the parameters to be complex, otherwise use `'r'`.

* Covariance-PCA Layer

	```
	cnn = cnnAddCoPCALayer(NN_NAME, FILTER_DIM, PCA_DIM, CORR_STEP, CORR_TYPE);
	```
	** FILTER_DIM: dimension of the covariance window.
	** PCA_DIM: the dimension of PCA.
	** CORR_STEP: The stride of correlation.
	** CORR_TYPE: `1` for auto-correlation and `2` for cross-correlation.

	Example:
	```
	cnn = cnnAddCoPCALayer(cnn, [3, 3], 1:3, [2, 2], 1);
	% Add a CoPCA layer with correlation window 3*3, stide 2*2, PCA keeps the larget three singular values, and do auto-correlation.
	```

* Transform Layer

	Add a layer which do transform to the input data. Syntax:

	```
	cnn = cnnAddTransformLayer(NN_NAME, TRAMSFORM, ...)
	```

	Currently supported transforms:
	* `'pca'`
	
	PCA Layer. Use syntax `cnnAddTransformLayer(NN_NAME, 'pca', PCA_DIM);` to specify the PCA dimension. 

	* `'fft'` 
	
	FFT Layer.

	* `dwt`

	DWT Layer. Use syntax `cnnAddTransformLayer(NN_NAME, 'dwt', WAVELET_NAME);` to specify the wavelet name (must be supported by MATLAB Wavelet Toolbox). 
	
	*  `abs` `arg` `real` `imag`
	
	ABS/ARG/REAL/IMAG Layer for complex input.
	
	See `cnnAddTransformLayer.m` for detail. 
	
More special SP layers and transformation types can be easilly added manually. 

See examples of Special SP layer:
>	`cnnAddCSLayer.m` `cnnCS.m`  
>	`cnnAddCoPCALayer.m` `cnnCoPCA.m`  
>	`cnnAddTransformLayer.m` `cnnTransform.m`

## Supported Training Method
* BP (Gradient descent)

Syntax:
	```
	to.test = 0;                % Flag for training or test
	[ERR, cnn] = cnnTrainBP(NN_NAME, TRAINING_DATA, TRAINING_LABEL);
	```

	** `ERR`: array contains training accuracies and costs.

## Validation

Syntax:
	```
	to.test = 1;                % Flag for training or test
	[acc, e] = cnnTestData(NN_NAME, TEST_DATA, TEST_LABEL, TEST_SIZE);
	```

	** `acc`: validation accuracy.
	** `e`: validation result array.
	** `TEST_SIZE`: use how many data to do the validation.


## Example
See the following file as an example of utilizing this toolbox:
>	`example_MNIST.m` 

Dataset used in this example is from [here](http://yann.lecun.com/exdb/mnist/).

See the following file as an example of utilization of Multiple channel (BLOB) layer and the insertion of SP Module:
>	`example_MicroDoppler.m`

Dataset used in this example is from [here](https://www.mathworks.com/help/phased/examples/pedestrian-and-bicyclist-classification-using-deep-learning.html?s_eid=PEP_16543).

You can download the MAT file used in this example from [here](https://1drv.ms/u/s!Akr-loZjbPYVufFK8J6pAMtHi1fEyA?e=T6Vgss).

## Roadmap
* [x] CPU Training support.
* [x] Batched Normalization Layer.
* [x] Dropout Layer.
* [x] MSE Output.
* [ ] LOWPASS/HIPASS filters in the Transform Layer.
* [ ] More training methods.
* [ ] RNN support.

## Reference

- [1] Zhang, Z., Jian, M., Lu, Z., Chen, H., James, S., Wang, C., & Gentile, R. (2020, April). Embedded Micro Radar for Pedestrian Detection in Clutter. In 2020 IEEE International Radar Conference (RADAR) (pp. 368-372). IEEE.

- [2] Zhang, Z., Chen, X., & Tian, Z. (2018, November). A hybrid neural network framework and application to radar automatic target recognition. In 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP) (pp. 246-250). IEEE.
