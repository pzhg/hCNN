function [dx, dgamma, dbeta]=cnnDeBatchedNormalization(BLayer, dout, x)

if BLayer.mode==1
    % CNN
    N=size(dout, 4);

    dout1=BLayer.gamma.*dout;
    x_norm=x-BLayer.cache.sample_mean;
    var_eps=BLayer.cache.sample_var+BLayer.eps;
    dvar=sum(dout1.*x_norm.*(-0.5).*var_eps.^(-1.5), [1, 2, 4]);
    % dx1=1./sqrt(BLayer.cache.sample_var+BLayer.eps);
    % dvar1=2*(x-BLayer.cache.sample_mean)/N;
    % 
    % di=dout1.*dx1+dvar.*dvar1;
    % dmean=-1*sum(di, 4);
    % dmean1=ones(size(x))/N;
    % 
    % dx=di+dmean.*dmean1;
    dmean=sum(dout1.*(-1./sqrt(var_eps)), [1, 2, 4])+dvar.*(-2).*sum(x_norm, [1, 2, 4])/N;    
    dx=dout1./sqrt(var_eps)+dmean/N+dvar.*2.*x_norm/N;
    dgamma=sum(dout.*BLayer.cache.out, [1, 2, 4]);
    dbeta=sum(dout, [1, 2, 4]);
else
    % FC
    N=size(dout, 2);

    dout1=BLayer.gamma.*dout;
    x_norm=x-BLayer.cache.sample_mean;
    var_eps=BLayer.cache.sample_var+BLayer.eps;
    dvar=sum(dout1.*x_norm.*(-0.5).*var_eps.^(-1.5), 2);
    % dx1=1./sqrt(BLayer.cache.sample_var+BLayer.eps);
    % dvar1=2*(x-BLayer.cache.sample_mean)/N;
    % 
    % di=dout1.*dx1+dvar.*dvar1;
    % dmean=-1*sum(di, 4);
    % dmean1=ones(size(x))/N;
    % 
    % dx=di+dmean.*dmean1;
    dmean=sum(dout1.*(-1./sqrt(var_eps)), 2)+dvar.*(-2).*sum(x_norm, 2)/N;    
    dx=dout1./sqrt(var_eps)+dmean/N+dvar.*2.*x_norm/N;
    dgamma=sum(dout.*BLayer.cache.out, 2);
    dbeta=sum(dout, 2);
end