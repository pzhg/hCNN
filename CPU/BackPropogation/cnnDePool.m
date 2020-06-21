function DeltaUnpool = cnnDePool(PLayer, DeltaPooled)

    numImages = size(DeltaPooled, 4);
    convDim = PLayer.OutDim .* PLayer.poolDim;
    numFilters = PLayer.FNum;

    DeltaUnpool = zeros(convDim(1), convDim(2), PLayer.FNum, numImages);

    switch PLayer.poolMethod
        case {'mean', 'MEAN', 'Mean'}
        % Mean Pooling
            parfor imNum = 1:numImages
                for filterNum = 1:numFilters
                    unpool = DeltaPooled(:, :, filterNum, imNum);
                    DeltaUnpool(:, :, filterNum, imNum) = kron(unpool, ones(PLayer.poolDim)) ./ (PLayer.poolDim(1) * PLayer.poolDim(2)) .* PLayer.poolLocation(:, :, filterNum, imNum);
                end
            end

        case {'max', 'MAX', 'Max'}
        % Max Pooling
            parfor imNum = 1:numImages
                for filterNum = 1:numFilters
                    temp = zeros(PLayer.poolDim(1) * PLayer.poolDim(2), PLayer.OutDim(1) * PLayer.OutDim(2));
                    m = reshape(DeltaPooled(:, :, filterNum, imNum), 1, PLayer.OutDim(1) * PLayer.OutDim(2));
                    i = sub2ind(size(temp), PLayer.poolLocation(1, :, filterNum, imNum), 1:PLayer.OutDim(1) * PLayer.OutDim(2));
                    temp(i) = m;
                    DeltaUnpool(:, :, filterNum, imNum) = col2im(temp, [PLayer.poolDim(1), PLayer.poolDim(2)], [convDim(1), convDim(2)], 'distinct');
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
        
        otherwise
            error('Unknown Pooling Method!');
    end

end