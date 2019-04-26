function cnn=cnnAddCSLayer(cnn, OutDim)
% Input Layer
%   inputSize: dimensionality of input data, [x-dim, y-dim]
%   channelNum: number of channels of input data

CLayer=struct;
CLayer.type=101;
CLayer.FDim=OutDim;
CLayer.OutDim=OutDim*cnn.Layers{cnn.LNum}.FNum;
CLayer.A=gpuArray.randn(CLayer.FDim, cnn.Layers{cnn.LNum}.OutDim(1)*cnn.Layers{cnn.LNum}.OutDim(2));
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=CLayer;