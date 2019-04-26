function Features=cnnFeedForwardBLOB(BLayer, images)

for inet=1:BLayer.NNum
    tcnn=BLayer.Nets{inet};
    BLayer.OutDim=BLayer.OutDim+tcnn.Layers{cnn.LNum}.OutDim;
end
numImages=size(images, 4);
numFilters=CLayer.FNum;
convolvedFeatures=gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages);

parfor i=1:numImages
    for fil2=1:numFilters
        convolvedImage=gpuArray.zeros(CLayer.OutDim);
        for fil1=1:numFilters1
            filter=rot90(squeeze(CLayer.W(:, :, fil1, fil2)), 2);
            im=squeeze(images(:, :, fil1, i));
            convolvedImage=convolvedImage+conv2(im, filter, 'valid');
        end
        convolvedImage=bsxfun(@plus, convolvedImage, gpuArray(CLayer.B(fil2)));
        convolvedFeatures(:, :, fil2, i)=convolvedImage;
    end
end