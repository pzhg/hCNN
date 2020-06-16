function [Wc_grad, bc_grad]=cnnConvGrad_GPU(activationsPooled, DeltaConv)

numImages=size(activationsPooled, 4);
numFilters2=size(DeltaConv, 3);
numFilters1=size(activationsPooled, 3);
ConvDim=size(activationsPooled)-size(DeltaConv)+1;

Wc_grad=zeros(ConvDim(1), ConvDim(2), numFilters1, numFilters2, 'single');
bc_grad=zeros(numFilters2, 1, 'single');
parfor fil2=1:numFilters2
    for fil1=1:numFilters1
        for im=1:numImages
            Wc_grad(:, :, fil1, fil2)=Wc_grad(:, :, fil1, fil2)+gather(conv2(gpuArray(activationsPooled(:, :, fil1, im)), gpuArray(rot90(DeltaConv(:, :, fil2, im), 2)), 'valid'));
        end
    end
    temp=DeltaConv(:, :, fil2, :);
    bc_grad(fil2)=sum(temp(:));
%     clear temp;
end