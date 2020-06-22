function out = cnnTransform(images, CLayer)

    switch CLayer.TName
        case {'FFT', 'fft'}
            out = fft(fft(images, [], 1), [], 2);
        case {'DWT', 'dwt'}
            numImage = size(images, 4);
            numFilter = size(images, 3);
            out_ = zeros(CLayer.OutDim(1), CLayer.OutDim(2), numFilter, 4, numImage);

            parfor inum = 1:numImage
                for iflt = 1:numFilter
                    wOut = zeros(CLayer.OutDim(1), CLayer.OutDim(2), 4);
                    [wOut(:, :, 1), wOut(:, :, 2), wOut(:, :, 3), wOut(:, :, 4)] = dwt2(images(:, :, iflt, inum));
                    out_(:, :, iflt, :, inum) = wOut;
                    % out_(:, :, (iflt - 1) * 4 + 2, inum) = LH;
                    % out_(:, :, (iflt - 1) * 4 + 3, inum) = HL;
                    % out_(:, :, (iflt - 1) * 4 + 4, inum) = HH;
                end
            end

            out = reshape(out_, CLayer.OutDim(1), CLayer.OutDim(2), numFilter * 4, numImage);
        case {'PCA', 'pca'}
            numImage = size(images, 4);
            numFilter = size(images, 3);
            out = zeros(size(images));

            parfor inum = 1:numImage
                for iflt = 1:numFilter
                    [U, S, V] = svd(double(images(:, :, iflt, inum)));
                    U = U(:, CLayer.PCADim);
                    S = S(CLayer.PCADim, :);
                    PCAImage = U * S * V';
                    out(:, :, iflt, inum) = PCAImage;
                end
            end

        case {'ABS', 'abs'}
            out = abs(images);
        case {'ARG', 'arg'}
            out = angle(images);
        case {'REAL', 'real'}
            out = real(images);
        case {'IMAG', 'imag'}
            out = imag(images);
        case {'MAXPOOL', 'maxpool'}
            CLayer.poolMethod = 'max';
            [~, out] = cnnPool(CLayer, images);
        case {'MEANPOOL', 'meanpool'}
            CLayer.poolMethod = 'mean';
            [~, out] = cnnPool(CLayer, images);
        case {'LOWPASS', 'lowpass'}
        case {'HIGHPASS', 'highpass'}
        otherwise
            error('Unknown Transformation Type!');
    end

end