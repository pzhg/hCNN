function [DeBNFeatrures, dgamma, dbeta]=cnnDeBatchedNormalization(BLayer, DeltaBN, images, BNimages)

numImages=size(DeltaBN, 4);
sample_mean=mean(images, 4);
sample_var=var(images, 0, 4);

deOut1=BLayer.gamma*DeltaBN;
deVar=sum(deOut*(images-sample_mean)*(-0.5)*(sample_var+BLayer.eps)*(-1.5), 4);
deImage1=1/sqrt(sample_var+eps);
deVar1=2*(images-sample_mean)/numImages;

di=deOut1*deImage1+deVar*deVar1;
deMean=-1*sum(di, 4);
deMean1=ones(size(images))/numImages;

DeBNFeatrures=di+deMean*deMean1;
dgamma=sum(deOut*BNimages, 4);
dbeta=sum(deOut, 1);