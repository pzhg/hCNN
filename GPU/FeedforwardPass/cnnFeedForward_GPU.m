function cnn = cnnFeedForward_GPU(cnn, images)

    numImages = size(images, 4);
    cnn.wCost = 0;
    cnn.Delta = cell(1, cnn.LNum + 1);
    cnn.OutData = cell(1, cnn.LNum);

    for iLayer = 1:cnn.LNum

        switch cnn.Layers{iLayer}.type
            case 0
                % Input Layer
                cnn.OutData{iLayer} = images;

                if length(cnn.Layers{1}.OutDim) == 1
                    cnn.OutData{iLayer} = squeeze(cnn.OutData{iLayer});
                end

            case 1
                % Hybrid Convolution Layer
                cnn.OutData{iLayer} = cnnConvolveRadar_GPU(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1});
            case 2
                % Convolution Layers
                cnn.wCost = cnn.wCost + sum(cnn.Layers{iLayer}.W(:).^2);
                cnn.OutData{iLayer} = cnnConvolve_GPU(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1});
            case 3
                % Fully Connected Layers
                cnn.wCost = cnn.wCost + sum(cnn.Layers{iLayer}.W(:).^2);
                cnn.OutData{iLayer} = cnnFullConnected_GPU(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1});
            case 4
                % Softmax Layer
                cnn.OutData{iLayer} = cnnSoftMax(cnn.OutData{iLayer - 1});
            case 5
                % Pooling Layers
                [cnn.Layers{iLayer}, cnn.OutData{iLayer}] = cnnPool_GPU(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1});
            case 6
                % Reshape Layer
                %             Data=gpuArray.zeros();
                cnn.OutData{iLayer} = reshape(cnn.OutData{iLayer - 1}, [], numImages);
            case 7
                % Activation Function Layer
                cnn.OutData{iLayer} = cnnActivate(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1});
            case 8
                % RMSE Layer
                cnn.OutData{iLayer} = cnn.OutData{iLayer - 1};
            case 9
                % (Deprecated) SP Filter Layer
                cnn.OutData{iLayer} = gpuArray.zeros(cnn.Layers{iLayer}.OutDim, numImages, 'single');
                cnn.OutData{iLayer}(1:cnn.Layers{iLayer - 1}.OutDim, :) = cnn.OutData{iLayer - 1};
                %             OutData{iLayer}(cnn.Layers{iLayer-1}.OutDim+1:cnn.Layers{iLayer}.OutDim, :)=OptData;
            case 10
                % BLOB Layer
                cnn.OutData{iLayer} = gpuArray.zeros(cnn.Layers{iLayer}.OutDim, numImages, 'single');

                if cnn.Layers{iLayer}.combineType == 1
                    %             offset=0;
                    for inet = 1:cnn.Layers{iLayer}.NNum
                        tcnn = cnn.Layers{iLayer}.Nets{inet};
                        tcnn = cnnFeedForward_GPU(tcnn, cnn.OutData{iLayer - 1});
                        %                 cnn.OutData{iLayer}(offset+1:offset+tcnn.Layers{tcnn.LNum}.OutDim, :)=tcnn.OutData{tcnn.LNum};
                        %                 offset=offset+tcnn.Layers{tcnn.LNum}.OutDim;
                        cnn.OutData{iLayer} = cnn.OutData{iLayer} + tcnn.OutData{tcnn.LNum};
                        cnn.Layers{iLayer}.Nets{inet} = tcnn;
                    end

                else
                    offset = 0;

                    for inet = 1:cnn.Layers{iLayer}.NNum
                        tcnn = cnn.Layers{iLayer}.Nets{inet};
                        tcnn = cnnFeedForward_GPU(tcnn, cnn.OutData{iLayer - 1});
                        cnn.OutData{iLayer}(offset + 1:offset + tcnn.Layers{tcnn.LNum}.OutDim, :) = tcnn.OutData{tcnn.LNum};
                        offset = offset + tcnn.Layers{tcnn.LNum}.OutDim;
                        cnn.Layers{iLayer}.Nets{inet} = tcnn;
                    end

                end

            case 101
                % CS
                %             if cnn.Layers{iLayer}.calced==0
                cnn.OutData{iLayer} = cnnCS_GPU(cnn.OutData{iLayer - 1}, cnn.Layers{iLayer});
                %                 cnn.Layers{iLayer}.calced=1;
                %             end
            case 102
                % CoPCA
                %             if cnn.Layers{iLayer}.calced==0
                cnn.OutData{iLayer} = cnnCoPCA_GPU(cnn.OutData{iLayer - 1}, cnn.Layers{iLayer});
                %                 cnn.Layers{iLayer}.calced=1;
                %             end
            case 103
                % End
                cnn.OutData{iLayer} = cnn.OutData{iLayer - 1};
            case 104
                % Transform
                cnn.OutData{iLayer} = cnnTransform_GPU(cnn.OutData{iLayer - 1}, cnn.Layers{iLayer});
            case 11
                % Batched Normalization
                [cnn.OutData{iLayer}, cnn.Layers{iLayer}] = cnnBatchedNormalization(cnn.Layers{iLayer}, cnn.OutData{iLayer - 1}, cnn.to.test);
            case 12
                % Dropout Layer
                [cnn.OutData{iLayer}, cnn.Layers{iLayer}] = cnnDropOut(cnn.OutData{iLayer - 1}, cnn.Layers{iLayer});
            otherwise
                error('Unknown Layer Type!');
        end

    end

end
