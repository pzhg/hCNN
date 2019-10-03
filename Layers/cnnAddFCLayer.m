function cnn=cnnAddFCLayer(cnn, OutDim, comp, DropOut)
% Fully Connected Layer
%   OutDim: number of nuerons

FLayer=struct;
FLayer.type=3;
FLayer.OutDim=OutDim;
FLayer.W=single(0.1*gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim));
FLayer.B=single(0.1*gpuArray.zeros(OutDim, 1));
if comp=='c'
    FLayer.W=single(FLayer.W+0.1*1j*gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim));
    FLayer.B=single(FLayer.B+0.1*1j*gpuArray.zeros(OutDim, 1));
end
FLayer.FNum=1;
FLayer.DropOut=DropOut;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=FLayer;