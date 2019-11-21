function cnn=cnnAddFCLayer(cnn, OutDim, comp, DropOut)
% Fully Connected Layer
%   OutDim: number of nuerons

FLayer=struct;
FLayer.type=3;
FLayer.OutDim=OutDim;
FLayer.W=single(0.1*randn(OutDim, cnn.Layers{cnn.LNum}.OutDim));
FLayer.B=single(zeros(OutDim, 1));
if comp=='c'
    FLayer.W=single(FLayer.W+0.1*1j*randn(OutDim, cnn.Layers{cnn.LNum}.OutDim));
    FLayer.B=single(FLayer.B+1j*zeros(OutDim, 1));
end
FLayer.FNum=1;
FLayer.DropOut=DropOut;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=FLayer;