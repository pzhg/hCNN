function fltOutput=cnnCS(images, CLayer)

numImages=size(images, 4);
numFilter=size(images, 3);
fltOutput=gpuArray.zeros(CLayer.OutDim, numImages);

for inum=1:numImages
    for iflt=1:numFilter
        image=images(:, :, iflt, inum);
        CSImage=CLayer.A*image(:);
        fltOutput((iflt-1)*CLayer.FDim+1:iflt*CLayer.FDim, inum)=CSImage;
    end
end