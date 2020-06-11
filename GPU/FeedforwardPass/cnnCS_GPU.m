function fltOutput=cnnCS(images, CLayer)

numImages=size(images, 4);
numFilter=size(images, 3);
if CLayer.useGPU==1
    fltOutput=single(gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), numFilter, numImages));
else
    fltOutput=single(zeros(CLayer.OutDim(1), CLayer.OutDim(2), numFilter, numImages));
end

for inum=1:numImages
    for iflt=1:numFilter
        image=images(:, :, iflt, inum);
        CSImage=single(CLayer.A*image(:));
        fltOutput(:, 1, iflt, inum)=CSImage;
    end
end