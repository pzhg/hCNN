function cnn=cnnAddSoftMaxLayer(cnn)
% SoftMax Output Layer

SLayer.type=8;
SLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim;
SLayer.FNum=1;
cnn.outputDim=SLayer.OutDim;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=SLayer;