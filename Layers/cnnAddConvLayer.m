function cnn=cnnAddConvLayer(cnn, FDim, FNum, comp)
% Convolutional Layer
%   FDim: filter dimensionality, [x-dim, y-dim]
%   FNum: filter number

CLayer=struct;
CLayer.type=2;
CLayer.FDim=FDim;
CLayer.FNum=FNum;
CLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim-FDim+1;
if cnn.to.useGPU==1
    CLayer.W=single(0.1*gpuArray.randn(FDim(1), FDim(2), cnn.Layers{cnn.LNum}.FNum, FNum));
    if comp=='c'
        CLayer.W=single(CLayer.W+0.1*1j*gpuArray.randn(FDim(1), FDim(2), cnn.Layers{cnn.LNum}.FNum, FNum));
    end
    CLayer.B=single(gpuArray.zeros(FNum, FNum));
else
    CLayer.W=single(0.1*randn(FDim(1), FDim(2), cnn.Layers{cnn.LNum}.FNum, FNum));
    if comp=='c'
        CLayer.W=single(CLayer.W+0.1*1j*randn(FDim(1), FDim(2), cnn.Layers{cnn.LNum}.FNum, FNum));
    end
    CLayer.B=single(zeros(FNum, FNum));
end
CLayer.useGPU=cnn.to.useGPU;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=CLayer;


