function [out, BLayer]=cnnBatchedFilter(BLayer, x, test)

if test==0
    if BLayer.mode==1
    % CNN
        BLayer.cache.sample_mean=gather(mean(x, [1, 2, 4]));
        BLayer.cache.sample_var=gather(var(x, 0, [1, 2, 4]));
        BLayer.cache.out=gather((x-BLayer.cache.sample_mean)./sqrt(BLayer.cache.sample_var+BLayer.eps));
        BLayer.mean=gather(BLayer.mean*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_mean);
        BLayer.var=gather(BLayer.var*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_var);
        out=BLayer.gamma.*BLayer.cache.out+BLayer.beta;
    else
    % FC
        BLayer.cache.sample_mean=gather(mean(x, 2));
        BLayer.cache.sample_var=gather(var(x, 0, 2));
        BLayer.cache.out=gather((x-BLayer.cache.sample_mean)./sqrt(BLayer.cache.sample_var+BLayer.eps));
        BLayer.mean=gather(BLayer.mean*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_mean);
        BLayer.var=gather(BLayer.var*BLayer.mom+(1-BLayer.mom)*BLayer.cache.sample_var);
        out=BLayer.gamma.*BLayer.cache.out+BLayer.beta;
    end
else
    scale=BLayer.gamma./sqrt(BLayer.var+BLayer.eps);
    out=x.*scale+(BLayer.beta-BLayer.mean.*scale);
end
