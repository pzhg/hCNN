function convolvedFeatures = cnnConvolveRadar_GPU(RLayer, images)

    numImages = size(images, 4);

    convolvedFeatures = gpuArray.zeros(RLayer.OutDim(1), RLayer.OutDim(2), 1, numImages, 'single');

    HFSize = floor((RLayer.FDim - 1) ./ 2);
    XFilter = gpuArray(single(exp(1j * pi * RLayer.Ka * (((-HFSize:HFSize).' / RLayer.PRF).^2)) * exp(-1j * pi * RLayer.Kr * (((-HFSize:HFSize) / RLayer.Fsr).^2))));

    parfor i_img = 1:numImages
        convolvedImage = gpuArray.zeros(RLayer.OutDim, 'single');
        im = images(:, :, :, i_img);
        convolvedImage = convolvedImage + conv2(im, XFilter, 'valid');
        convolvedFeatures(:, :, 1, i_img) = convolvedImage;
    end

end