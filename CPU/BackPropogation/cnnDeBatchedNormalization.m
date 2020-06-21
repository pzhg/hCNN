function [dx, dgamma, dbeta] = cnnDeBatchedNormalization(BLayer, dout)

    switch BLayer.mode
        case 1
            % CNN
            N = size(dout, 1) * size(dout, 2) * size(dout, 4);

            dout_ = BLayer.gamma .* dout;
            %     x_norm=x-BLayer.cache.sample_mean;
            var_eps = BLayer.cache.sample_var + BLayer.eps;
            %     dvar=sum(dout_.*x_norm.*(-0.5).*var_eps.^(-1.5), [1, 2, 4]);
            %     dx_=1./sqrt(var_eps);
            %     dvar_=2*x_norm/N;
            %     di=dout_.*dx_+dvar.*dvar_;
            %     dmean=-1*sum(di, [1, 2, 4]);
            %     dmean_=ones(size(x))/N;
            %     dx=di+dmean.*dmean_;

            %     dmean=sum(dout_.*(-1./sqrt(var_eps)), [1, 2, 4])+dvar.*(-2).*sum(x_norm, [1, 2, 4])/N;
            %     dx=dout_./sqrt(var_eps)+dmean/N+dvar.*2.*x_norm/N;

            dx = 1 ./ (N .* sqrt(var_eps)) .* (dout_ .* N - sum(dout_, [1, 2, 4]) - BLayer.cache.out .* sum(dout_ .* BLayer.cache.out, [1, 2, 4]));
            dgamma = sum(dout .* BLayer.cache.out, [1, 2, 4]);
            dbeta = sum(dout, [1, 2, 4]);

        case 2
            % FC
            N = size(dout, 2);

            dout_ = BLayer.gamma .* dout;
            %     x_norm=x-BLayer.cache.sample_mean;
            var_eps = BLayer.cache.sample_var + BLayer.eps;

            %     dvar=sum(dout_.*x_norm.*(-0.5).*var_eps.^(-1.5), 2);
            % dx1=1./sqrt(BLayer.cache.sample_var+BLayer.eps);
            % dvar1=2*(x-BLayer.cache.sample_mean)/N;
            % di=dout1.*dx1+dvar.*dvar1;
            % dmean=-1*sum(di, 4);
            % dmean1=ones(size(x))/N;
            % dx=di+dmean.*dmean1;

            %     dmean=sum(dout_.*(-1./sqrt(var_eps)), 2)+dvar.*(-2).*sum(x_norm, 2)/N;
            %     dx=dout_./sqrt(var_eps)+dmean/N+dvar.*2.*x_norm/N;

            dx = 1 ./ (N .* sqrt(var_eps)) .* (dout_ .* N - sum(dout_, 2) - BLayer.cache.out .* sum(dout_ .* BLayer.cache.out, 2));
            dgamma = sum(dout .* BLayer.cache.out, 2);
            dbeta = sum(dout, 2);

        otherwise
            error('Unknows Batched Normalization Layer Mode!');

    end

end