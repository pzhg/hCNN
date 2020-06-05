function cnn=cnnAddPoolLayer(cnn, poolMethod, poolDim)
% Pooling Layer
%   poolMethod: 'mean' -- mean pooling
%               'max'  -- max pooling
%   poolDim:    [x-dimension, y-dimension]

PLayer=struct;
PLayer.poolMethod=poolMethod;
PLayer.type=5;
PLayer.poolDim=poolDim;
if cnn.to.useGPU==1
    PLayer.poolLocation=gpuArray([]);
else
    PLayer.poolLocation=[];
end
PLayer.OutDim=floor(cnn.Layers{cnn.LNum}.OutDim./poolDim);
PLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
PLayer.useGPU=cnn.to.useGPU;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=PLayer;