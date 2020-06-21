function [out, BLayer] = cnnBatchedNormalization(BLayer, x, test)

    if test == 0

        switch BLayer.mode
            case 1
                % CNN
                BLayer.cache.sample_mean = mean(x, [1, 2, 4]);
                BLayer.cache.sample_var = var(x, 0, [1, 2, 4]);
                BLayer.cache.out = (x - BLayer.cache.sample_mean) ./ sqrt(BLayer.cache.sample_var + BLayer.eps);
                BLayer.mean = BLayer.mean * BLayer.mom + (1 - BLayer.mom) * BLayer.cache.sample_mean;
                BLayer.var = BLayer.var * BLayer.mom + (1 - BLayer.mom) * BLayer.cache.sample_var;
                out = BLayer.gamma .* BLayer.cache.out + BLayer.beta;
            case 2
                % FC
                BLayer.cache.sample_mean = mean(x, 2);
                BLayer.cache.sample_var = var(x, 0, 2);
                BLayer.cache.out = (x - BLayer.cache.sample_mean) ./ sqrt(BLayer.cache.sample_var + BLayer.eps);
                BLayer.mean = BLayer.mean * BLayer.mom + (1 - BLayer.mom) * BLayer.cache.sample_mean;
                BLayer.var = BLayer.var * BLayer.mom + (1 - BLayer.mom) * BLayer.cache.sample_var;
                out = BLayer.gamma .* BLayer.cache.out + BLayer.beta;
            otherwise
                error('Unknown Batched Normalization Layer Mode!');
        end

    else
        scale = BLayer.gamma ./ sqrt(BLayer.var + BLayer.eps);
        out = x .* scale + (BLayer.beta - BLayer.mean .* scale);
        
    end

end