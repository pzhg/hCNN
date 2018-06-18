function cnn=cnnAddFCLayer(cnn, OutDim)
% Fully Connected Layer
%   OutDim: number of nuerons

FLayer=struct;
FLayer.type=3;
FLayer.OutDim=OutDim;
FLayer.W=gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim);
FLayer.B=gpuArray.zeros(OutDim, 1);
if comp=='c'
    FLayer.W=FLayer.W+1j*gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim);
    FLayer.B=FLayer.B+1j*gpuArray.zeros(OutDim, 1);
end
FLayer.FNum=1;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=FLayer;