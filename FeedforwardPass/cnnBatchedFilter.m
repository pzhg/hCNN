function [out, BLayer]=cnnBatchedFilter(BLayer, x, test)

if test==0
    if BLayer.mode==1
    % CNN
        BLayer.cache.sample_mean=mean(x, [1, 2, 4]);
        BLayer.cache.sample_var=var(x, 0, [1, 2, 4]);
        BLayer.cache.out=(x-BLayer.cache.sample_mean)./(BLayer.cache.sample_var+BLayer.eps);
        BLayer.mean=BLayer.mean*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_mean;
        BLayer.var=BLayer.var*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_var;
        out=BLayer.gamma.*BLayer.cache.out+BLayer.beta;
    else
    % FC
        BLayer.cache.sample_mean=mean(x, 2);
        BLayer.cache.sample_var=var(x, 0, 2);
        BLayer.cache.out=(x-BLayer.cache.sample_mean)./(BLayer.cache.sample_var+BLayer.eps);
        BLayer.mean=BLayer.mean*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_mean;
        BLayer.var=BLayer.var*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_var;
        out=BLayer.gamma.*BLayer.cache.out+BLayer.beta;
    end
else
    scale=BLayer.gamma./sqrt(BLayer.var+BLayer.eps);
    out=x.*scale+(BLayer.beta-BLayer.mean.*scale);
end
