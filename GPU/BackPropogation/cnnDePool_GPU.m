function DeltaUnpool=cnnDePool(PLayer, DeltaPooled)

numImages=size(DeltaPooled, 4);
convDim=PLayer.OutDim.*PLayer.poolDim;
numFilters=PLayer.FNum;

if PLayer.useGPU==1
    DeltaUnpool=single(gpuArray.zeros(convDim(1), convDim(2), PLayer.FNum, numImages));
else
    DeltaUnpool=single(zeros(convDim(1), convDim(2), PLayer.FNum, numImages));
end        
switch PLayer.poolMethod
    case 'mean'
        parfor imNum=1:numImages
            for filterNum=1:numFilters
                unpool=DeltaPooled(:, :, filterNum, imNum);
                if PLayer.useGPU==1
                    DeltaUnpool(:, :, filterNum, imNum)=kron(unpool, single(gpuArray.ones(PLayer.poolDim)))./(PLayer.poolDim(1)*PLayer.poolDim(2)).*PLayer.poolLocation(:, :, filterNum, imNum);
                else
                    DeltaUnpool(:, :, filterNum, imNum)=kron(unpool, single(ones(PLayer.poolDim)))./(PLayer.poolDim(1)*PLayer.poolDim(2)).*PLayer.poolLocation(:, :, filterNum, imNum);
                end
%               clear unpool;
            end
        end
    case 'max'
        for imNum=1:numImages
            for filterNum=1:numFilters
                if PLayer.useGPU==1 
                    temp=single(gpuArray.zeros(PLayer.poolDim(1)*PLayer.poolDim(2), PLayer.OutDim(1)*PLayer.OutDim(2)));
                    m=reshape(DeltaPooled(:, :, filterNum, imNum), 1, PLayer.OutDim(1)*PLayer.OutDim(2));
                    i=sub2ind(size(temp),PLayer.poolLocation(1, :, filterNum, imNum), 1:PLayer.OutDim(1)*PLayer.OutDim(2));
                    temp(i)=m;
                    DeltaUnpool(:, :, filterNum, imNum)=single(gpuArray(col2im(gather(temp), [PLayer.poolDim(1), PLayer.poolDim(2)], [convDim(1), convDim(2)], 'distinct')));
                else
                    temp=single(zeros(PLayer.poolDim(1)*PLayer.poolDim(2), PLayer.OutDim(1)*PLayer.OutDim(2)));
                    m=reshape(DeltaPooled(:, :, filterNum, imNum), 1, PLayer.OutDim(1)*PLayer.OutDim(2));
                    i=sub2ind(size(temp),PLayer.poolLocation(1, :, filterNum, imNum), 1:PLayer.OutDim(1)*PLayer.OutDim(2));
                    temp(i)=m;
                    DeltaUnpool(:, :, filterNum, imNum)=single(col2im(gather(temp), [PLayer.poolDim(1), PLayer.poolDim(2)], [convDim(1), convDim(2)], 'distinct'));
                end
%                 ckear temp, m;
%                 for idx_j=1:PLayer.OutDim(1)
%                     for idx_i=1:PLayer.OutDim(2)
%                         startX=(idx_i-1)*PLayer.poolDim(1)+1;
%                         startY=(idx_j-1)*PLayer.poolDim(2)+1;
%                         poolField=gpuArray.zeros(1, PLayer.poolDim(1)*PLayer.poolDim(2));
%                         poolField(PLayer.poolLocation(idx_i, idx_j, filterNum, imNum))=DeltaPooled(idx_i, idx_j, filterNum, imNum);
%                         DeltaUnpool(startX:startX+PLayer.poolDim(1)-1, startY:startY+PLayer.poolDim(2)-1, filterNum, imNum)=reshape(poolField, PLayer.poolDim(1), PLayer.poolDim(2));
%                     end
%                 end
            end
        end
end