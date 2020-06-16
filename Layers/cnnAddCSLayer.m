function cnn=cnnAddCSLayer(cnn, OutDim)
% Input Layer
%   inputSize: dimensionality of input data, [x-dim, y-dim]
%   channelNum: number of channels of input data

CLayer=struct;
CLayer.type=101;
CLayer.calced=0;
CLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
% CLayer.FDim=OutDim;
CLayer.OutDim=[OutDim, 1];
% CLayer.useGPU=cnn.to.useGPU;
if cnn.to.useGPU==1
    CLayer.A=randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1)*cnn.Layers{cnn.LNum}.OutDim(2), 'single');
else
    CLayer.A=randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1)*cnn.Layers{cnn.LNum}.OutDim(2));
end
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=CLayer;