function BNFeatrures=cnnBatchedFilter(BLayer, images, test)

BNFeatures=single(gpuArray.zeros(size(images)));

if test==0
    sample_mean=mean(images, 4);
    sample_var=var(images, 0, 4);
    BNFeaters=(images-sample_mean)./(sample_var+eps);

    BLayer.mean=BLayer.mean*BLayer.mom+(1-BLayer.mom)*sample_mean;
    BLayer.var=BLayer.var*BLayer.mom+(1-BLayer.mom)*sample_var;

    BNFeatures=BLayer.gamma*BNFeaters+BLayer.beta;
else
    scale=BLayer.gamma/sqrt(BLayer.var+eps);
    BNFeatures=images*scale+(BLayer.beta-BLayer.mean*scale);
end
