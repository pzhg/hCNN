function DeltaUnpool=cnnDePool(PLayer, DeltaPooled)

numImages=size(DeltaPooled, 4);
convDim=PLayer.OutDim.*PLayer.poolDim;
numFilters=PLayer.FNum;

DeltaUnpool=gpuArray.zeros(convDim(1), convDim(2), PLayer.FNum, numImages);
% switch PLayer.poolMethod
%     case 'mean'
        parfor imNum=1:numImages
            for filterNum=1:numFilters
                unpool=DeltaPooled(:, :, filterNum, imNum);
                DeltaUnpool(:, :, filterNum, imNum)=kron(unpool, gpuArray.ones(PLayer.poolDim))./(PLayer.poolDim(1)*PLayer.poolDim(2)).*PLayer.poolLocation(:, :, filterNum, imNum);
            end
        end
%     case 'max'
%         for imNum=1:numImages
%             for filterNum=1:numFilters 
%                 for idx_j=1:PLayer.OutDim
%                     for idx_i=1:PLayer.OutDim
%                         startX=(idx_i-1)*PLayer.poolDim+1;
%                         startY=(idx_j-1)*PLayer.poolDim+1;
%                         poolField=gpuArray.zeros(1, PLayer.poolDim*PLayer.poolDim);
%                         poolField(PLayer.poolLocation(idx_i, idx_j, filterNum, imNum))=DeltaPooled(idx_i, idx_j, filterNum, imNum);
%                         DeltaUnpool(startX:startX+PLayer.poolDim-1, startY:startY+PLayer.poolDim-1, filterNum, imNum)=reshape(poolField, PLayer.poolDim, PLayer.poolDim);
%                     end
%                 end
%             end
%         end
% end