function cnn=cnnAddRadarLayer(cnn, FDim, Fsr, PRF)
% Radar Data Layer
%   FDim: dimensionality of matched filter, [x-dim, y-dim]
%   Fsr: range direction sampling rate
%   PRF: pulse repetition frequency

RLayer=struct;
RLayer.type=1;
RLayer.Ka=50;
RLayer.Kr=50;
RLayer.FDim=FDim;
RLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim-FDim+1;
RLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
RLayer.Fsr=Fsr;
RLayer.PRF=PRF;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=RLayer;