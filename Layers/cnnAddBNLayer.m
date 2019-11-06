function cnn=cnnAddBNLayer(cnn)
% Convolutional Layer
%   FDim: filter dimensionality, [x-dim, y-dim]
%   FNum: filter number

BLayer=struct;
BLayer.type=11;
BLayer.eps=1e-5;
BLayer.gamma=randn;
BLayer.beta=randn;
BLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim;
BLayer.FDim=cnn.Layers{cnn.LNum}.FDim;
BLayer.mean=0;
BLayer.var=0;
BLayer.mom=0.9;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=BLayer;


