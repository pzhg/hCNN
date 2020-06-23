function convolvedFeatures = cnnConvolveRadar(RLayer, images)

    numImages = size(images, 4);

    convolvedFeatures = zeros(RLayer.OutDim(1), RLayer.OutDim(2), 1, numImages);

    HFSize = floor((RLayer.FDim - 1) ./ 2);
    XFilter = exp(1j * pi * RLayer.Ka * (((-HFSize:HFSize).' / RLayer.PRF).^2)) * exp(-1j * pi * RLayer.Kr * (((-HFSize:HFSize) / RLayer.Fsr).^2));

    parfor i_img = 1:numImages
        convolvedImage = zeros(RLayer.OutDim);
        im = images(:, :, 1, i_img);
        convolvedImage = convolvedImage + conv2(im, XFilter, 'valid');
        convolvedFeatures(:, :, 1, i_img) = convolvedImage;
    end

end