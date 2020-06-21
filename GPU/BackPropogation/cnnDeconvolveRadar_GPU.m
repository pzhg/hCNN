function [DeltaKa, DeltaKr] = cnnDeconvolveRadar_GPU(RLayer, Delta, images)

    numImages = size(images, 4);

    DeltaKa = gpuArray.zeros(RLayer.OutDim(1), RLayer.OutDim(2), numImages, 'single');
    DeltaKr = gpuArray.zeros(RLayer.OutDim(1), RLayer.OutDim(2), numImages, 'single');

    HFSize = floor((RLayer.FDim - 1) / 2);
    Theta = single(pi) * RLayer.Ka * (((-HFSize:HFSize).' / RLayer.PRF).^2 * gpuArray.ones(1, RLayer.FDim(1), 'single')) - single(pi) * RLayer.Kr * (gpuArray.ones(RLayer.FDim(2), 1, 'single') * single(pi) * RLayer.Kr * (((-HFSize:HFSize) / RLayer.Fsr).^2));
    FKaSin = single(pi) * ((-HFSize:HFSize).' / RLayer.PRF).^2 * gpuArray.ones(1, RLayer.FDim(1), 'single') .* sin(Theta);
    FKaCos = single(pi) * ((-HFSize:HFSize).' / RLayer.PRF).^2 * gpuArray.ones(1, RLayer.FDim(1), 'single') .* cos(Theta);
    FKrSin = gpuArray.ones(RLayer.FDim(2), 1, 'single') * single(pi) * (((-HFSize:HFSize) / RLayer.Fsr).^2) .* sin(Theta);
    FKrCos = gpuArray.ones(RLayer.FDim(2), 1, 'single') * single(pi) * (((-HFSize:HFSize) / RLayer.Fsr).^2) .* cos(Theta);

    parfor i_im = 1:numImages
        r_image = rot90(real(images(:, :, i_im)), 2);
        i_image = rot90(imag(images(:, :, i_im)), 2);
        DeltaKa(:, :, i_im) = real(Delta(:, :, :, i_im)) .* (conv2(r_image, FKaSin, 'valid') + conv2(i_image, FKaCos, 'valid')) + imag(Delta(:, :, :, i_im)) .* (conv2(i_image, FKaSin, 'valid') - conv2(r_image, FKaCos, 'valid'));
        DeltaKr(:, :, i_im) = real(Delta(:, :, :, i_im)) .* (-conv2(r_image, FKrSin, 'valid') - conv2(i_image, FKrCos, 'valid')) + imag(Delta(:, :, :, i_im)) .* (-conv2(i_image, FKrSin, 'valid') + conv2(r_image, FKrCos, 'valid'));
    end

end