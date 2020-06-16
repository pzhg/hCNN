function cnn=cnnAddFCLayer(cnn, OutDim, comp)
% Fully Connected Layer
%   OutDim: number of nuerons

FLayer=struct;
FLayer.type=3;
FLayer.OutDim=OutDim;
if cnn.to.useGPU==0
    FLayer.W=0.1*randn(OutDim, cnn.Layers{cnn.LNum}.OutDim);
    FLayer.B=zeros(OutDim, 1);
    if comp=='c'
        FLayer.W=FLayer.W+0.1*1j*randn(OutDim, cnn.Layers{cnn.LNum}.OutDim);
        FLayer.B=FLayer.B+1j*zeros(OutDim, 1);
    end
else
    FLayer.W=0.1*gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim, 'single');
    FLayer.B=gpuArray.zeros(OutDim, 1, 'single');
    if comp=='c'
        FLayer.W=FLayer.W+0.1*1j*gpuArray.randn(OutDim, cnn.Layers{cnn.LNum}.OutDim, 'single');
        FLayer.B=FLayer.B+1j*gpuArray.zeros(OutDim, 1, 'single');
    end
end
FLayer.FNum=1;
% FLayer.DropOut=DropOut;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=FLayer;