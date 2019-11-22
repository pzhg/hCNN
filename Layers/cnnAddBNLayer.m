function cnn=cnnAddBNLayer(cnn, mode)
% Convolutional Layer
%   FDim: filter dimensionality, [x-dim, y-dim]
%   FNum: filter number

BLayer=struct;
BLayer.type=11;
BLayer.eps=1e-6;
BLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim;
BLayer.mode=mode;
BLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
if BLayer.mode==1
    % CNN
    BLayer.FDim=cnn.Layers{cnn.LNum}.FDim;
    BLayer.gamma=single(ones(1, 1, BLayer.FNum, 1));
    BLayer.beta=single(zeros(1, 1, BLayer.FNum, 1));
    BLayer.mean=single(zeros(1, 1, BLayer.FNum, 1));
    BLayer.var=single(zeros(1, 1, BLayer.FNum, 1));
else
    % FC
    BLayer.gamma=single(ones(BLayer.OutDim, 1));
    BLayer.beta=single(zeros(BLayer.OutDim, 1));
    BLayer.mean=single(zeros(BLayer.OutDim, 1));
    BLayer.var=single(zeros(BLayer.OutDim, 1));
end
BLayer.mom=0.9;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=BLayer;


