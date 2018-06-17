function [DeltaKa, DeltaKr]=cnnDeconvolveRadar(RLayer, Delta, images)

numImages=size(images, 4);
DeltaKa=gpuArray.zeros(RLayer.OutDim(1), RLayer.OutDim(2), numImages);
DeltaKr=gpuArray.zeros(RLayer.OutDim(1), RLayer.OutDim(2), numImages);

HFSize=floor((RLayer.FDim-1)/2);
FKaSin=gpuArray(pi*(([-HFSize:HFSize].'/RLayer.PRF).^2).*sin(pi*RLayer.Ka*(([-HFSize:HFSize].'/RLayer.PRF).^2)))*gpuArray.ones(1, RLayer.FDim(1));
FKaCos=gpuArray(pi*(([-HFSize:HFSize].'/RLayer.PRF).^2).*cos(pi*RLayer.Ka*(([-HFSize:HFSize].'/RLayer.PRF).^2)))*gpuArray.ones(1, RLayer.FDim(1));
FKrSin=gpuArray.ones(RLayer.FDim(2), 1)*gpuArray(pi*([-HFSize:HFSize]/RLayer.Fsr).^2).*sin(pi*RLayer.Kr*(([-HFSize:HFSize]/RLayer.Fsr).^2));
FKrCos=gpuArray.ones(RLayer.FDim(2), 1)*gpuArray(pi*([-HFSize:HFSize]/RLayer.Fsr).^2).*cos(pi*RLayer.Kr*(([-HFSize:HFSize]/RLayer.Fsr).^2));
parfor i_im=1:numImages
    r_image=rot90(gpuArray(squeeze(real(images(:, :, i_im)))), 2);
    i_image=rot90(gpuArray(squeeze(imag(images(:, :, i_im)))), 2);
    DeltaKa(:, :, i_im)=squeeze(real(Delta(:, :, :, i_im))).*(conv2(r_image, FKaSin, 'valid')+conv2(i_image, FKaCos, 'valid'))+squeeze(imag(Delta(:, :, :, i_im))).*(conv2(i_image, FKaSin, 'valid')-conv2(r_image, FKaCos, 'valid'));
    DeltaKr(:, :, i_im)=squeeze(real(Delta(:, :, :, i_im))).*(-conv2(r_image, FKrSin, 'valid')-conv2(i_image, FKrCos, 'valid'))+squeeze(imag(Delta(:, :, :, i_im))).*(-conv2(i_image, FKrSin, 'valid')+conv2(r_image, FKrCos, 'valid'));  
end