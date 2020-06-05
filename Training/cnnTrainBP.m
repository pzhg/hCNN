function [ERR, cnn]=cnnTrainBP(cnn, X, Y)
% Train CNN using BP gradient descend method
% Input:
%   X: training data, [x-dim, y-dim, channel-num, batch-size,
%                                batch-count]
%   Y: label data, [1, batch-size, batch-count]
%   to: training options
% Output:
%   ERR: array contains the accuracy and cost in each iteration
%   cnn: the trained CNN

ERR=[];
for e_count=1:cnn.to.epochs
    for b_count=1:cnn.to.batch
        %% Training Data
        mb_labels=squeeze(Y(:, :, b_count));
        if cnn.to.useGPU==1
            images=single(gpuArray(X(:, :, :, :, b_count)));
        else
            images=single(X(:, :, :, :, b_count));
        end
        numImages=cnn.to.batch_size;
        % Momemtum
        cnn.to.mom=single(cnn.to.mom);
        if b_count==cnn.to.momIncrease
            cnn.to.mom=single(cnn.to.momentum);
        end
%         if to.PCAflag==1
%             for iLayer=1:cnn.LNum
%                 if cnn.Layers{iLayer}.type==9
%                     fltLayer=iLayer;
%                     break;
%                 end
%             end
%             OptData=cnnFilter(images, cnn.Layers{fltLayer});
%         else
%             OptData=[];
%         end
           
        %% Feedforward Pass
        cnn=cnnFeedForward(cnn, images);
        
        %% Calculate Cost
        switch cnn.Layers{cnn.LNum}.type
            case 4
                index=sub2ind([cnn.outputDim, cnn.to.batch_size], squeeze(mb_labels), 1:cnn.to.batch_size);
                if cnn.to.useGPU==1
                    outPut=single(gpuArray.zeros(cnn.outputDim, cnn.to.batch_size));
                else
                    outPut=single(zeros(cnn.outputDim, cnn.to.batch_size));
                end
                outPut(index)=1;
                ceCost=-sum(sum(1e-6+log(cnn.OutData{cnn.LNum}(index))));
            case 8
                if cnn.to.useGPU==1
                    outPut=gpuArray(single(squeeze(mb_labels)));
                else
                    outPut=single(squeeze(mb_labels));
                end
                ceCost=1/2*sum((cnn.OutData{cnn.LNum}(:)-outPut(:)).^2);
        end
        wCost=cnn.to.lambda*cnn.wCost/2;
        cost=gather(ceCost)/numImages+wCost;
        
        %% BackPropagation
        cnn=cnnBackPropagation(cnn, outPut);

        %% Gradient Calculation and Update
        cnn=cnnUpdateWeight(cnn);
        
        %% Monitor Accuracy and Cost
        gpu=gpuDevice;
        switch cnn.Layers{cnn.LNum}.type
            case 4
                [~, preds]=max(cnn.OutData{cnn.LNum}, [], 1);
                acc=gather(sum(preds==mb_labels)/numImages);
                fprintf('Epoch %d: Cost on iteration %d is %f, accuracy is %f, avaliable memory is %f\n', e_count, b_count, cost, acc, gpu.AvailableMemory);
                ERR=[ERR, [cost; acc]];
            case 8
                fprintf('Epoch %d: Cost on iteration %d is %f\n', e_count, b_count, cost);
                ERR=[ERR, cost];
                
        end 
        reset(gpu);
        waitbar(((e_count-1)*cnn.to.batch+b_count)/(cnn.to.epochs*cnn.to.batch));
    end
    cnn.to.alpha=single(cnn.to.alpha)/single(2);
end