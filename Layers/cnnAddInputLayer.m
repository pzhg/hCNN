function cnn=cnnAddInputLayer(cnn, inputSize, channelNum)
% Input Layer
%   inputSize: dimensionality of input data, [x-dim, y-dim]
%   channelNum: number of channels of input data

ILayer=struct;
ILayer.type=0;
ILayer.OutDim=inputSize;
ILayer.FNum=channelNum;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=ILayer;