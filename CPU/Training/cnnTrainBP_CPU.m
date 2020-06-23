function [ERR, cnn] = cnnTrainBP_CPU(cnn, X, Y)
    % Train CNN using BP gradient descend method
    % Input:
    %   X: training data, [x-dim, y-dim, channel-num, batch-size,
    %                                batch-count]
    %   Y: label data, [1, batch-size, batch-count]
    % Output:
    %   ERR: array contains the accuracy and cost in each iteration
    %   cnn: the trained CNN

    ERR = [];

    for e_count = 1:cnn.to.epochs

        for b_count = 1:cnn.to.batch
            %% Training Data
            mb_labels = Y(:, :, b_count);
            images = X(:, :, :, :, b_count);
            numImages = cnn.to.batch_size;
            % Momemtum
            cnn.to.mom = cnn.to.mom;

            if b_count == cnn.to.momIncrease
                cnn.to.mom = cnn.to.momentum;
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
            cnn = cnnFeedForward(cnn, images);

            %% Calculate Cost
            switch cnn.Layers{cnn.LNum}.type
                case 4
                    index = sub2ind([cnn.outputDim, cnn.to.batch_size], squeeze(mb_labels), 1:cnn.to.batch_size);
                    outPut = zeros(cnn.outputDim, cnn.to.batch_size);
                    outPut(index) = 1;
                    ceCost = -sum(sum(1e-6 + log(cnn.OutData{cnn.LNum}(index))));
                case 8
                    outPut = squeeze(mb_labels);
                    ceCost = 1/2 * sum((cnn.OutData{cnn.LNum}(:) - outPut(:)).^2);
                otherwise
                    error('Illegal End Layer!');
            end

            wCost = cnn.to.lambda * cnn.wCost / 2;
            cost = ceCost / numImages + wCost;

            %% BackPropagation
            cnn = cnnBackPropagation(cnn, outPut);

            %% Gradient Calculation and Update
            cnn = cnnUpdateWeight(cnn);

            %% Monitor Accuracy and Cost
            switch cnn.Layers{cnn.LNum}.type
                case 4
                    [~, preds] = max(cnn.OutData{cnn.LNum}, [], 1);
                    acc = sum(preds == mb_labels) / numImages;
                    fprintf('Epoch %d: Cost on iteration %d is %f, accuracy is %f\n', e_count, b_count, cost, acc);
                    ERR = [ERR, [cost; acc]];
                case 8
                    fprintf('Epoch %d: Cost on iteration %d is %f\n', e_count, b_count, cost);
                    ERR = [ERR, cost];
                otherwise
                    error('Illegal Output Layer Type!');
            end

            %         reset(gpu);
            waitbar(((e_count - 1) * cnn.to.batch + b_count) / (cnn.to.epochs * cnn.to.batch));
        end

        cnn.to.alpha = single(cnn.to.alpha) / single(2);
    end

end