function cnn=cnnAddBLOBLayer(cnn, Nets, NNum, OutDim)

BLayer=struct;
BLayer.type=10;
BLayer.Nets=Nets;
BLayer.NNum=NNum;
BLayer.OutDim=OutDim;
% for inet=1:BLayer.NNum
%     tcnn=BLayer.Nets{inet};
%     BLayer.OutDim=BLayer.OutDim+tcnn.Layers{tcnn.LNum}.OutDim;
% end
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=BLayer;
