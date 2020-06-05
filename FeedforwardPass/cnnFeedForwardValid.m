function cnn=cnnFeedForwardValid(cnn, images)

numImages=size(images, 4);
cnn.wCost=0;
cnn.Delta=cell(1, cnn.LNum+1);
cnn.OutData=cell(1, cnn.LNum);
for iLayer=1:cnn.LNum
    switch cnn.Layers{iLayer}.type
        case 0
            % Input Layer
            if cnn.to.useGPU==1
                cnn.OutData{iLayer}=gpuArray(single(images));
            else
                cnn.OutData{iLayer}=single(images);
            end
            if size(cnn.Layers{iLayer}.OutDim, 2)==1
                cnn.OutData{iLayer}=squeeze(cnn.OutData{iLayer});
            end
        case 1
            % Hybrid Convolution Layer
            cnn.OutData{iLayer}=cnnConvolveRadar(cnn.Layers{iLayer}, cnn.OutData{iLayer-1});
        case 2
            % Convolution Layers
            cnn.wCost=cnn.wCost+sum(cnn.Layers{iLayer}.W(:).^2);
            cnn.OutData{iLayer}=cnnConvolve(cnn.Layers{iLayer}, cnn.OutData{iLayer-1});
        case 3
            % Fully Connected Layers
            cnn.wCost=cnn.wCost+sum(cnn.Layers{iLayer}.W(:).^2);
            cnn.OutData{iLayer}=cnnFullConnected(cnn.Layers{iLayer}, cnn.OutData{iLayer-1});
        case 4
            % Softmax Layer
            cnn.OutData{iLayer}=cnnSoftMax(cnn.OutData{iLayer-1});
        case 5
            % Pooling Layers
            [cnn.Layers{iLayer}, cnn.OutData{iLayer}]=cnnPool(cnn.Layers{iLayer}, cnn.OutData{iLayer-1});
        case 6
            % Reshape Layer
%             Data=gpuArray.zeros();
            cnn.OutData{iLayer}=reshape(cnn.OutData{iLayer-1}, [], numImages);
        case 7
            % Activation Function Layer
            cnn.OutData{iLayer}=cnnActivate(cnn.Layers{iLayer}, cnn.OutData{iLayer-1});
        case 8
            % RMSE Layer
            cnn.OutData{iLayer}=cnn.OutData{iLayer-1};
        case 9
            % (Deprecated) SP Filter Layer
            if cnn.to.useGPU==1
                cnn.OutData{iLayer}=single(gpuArray.zeros(cnn.Layers{iLayer}.OutDim, numImages));
            else
                cnn.OutData{iLayer}=single(zeros(cnn.Layers{iLayer}.OutDim, numImages));
            end
            cnn.OutData{iLayer}(1:cnn.Layers{iLayer-1}.OutDim, :)=cnn.OutData{iLayer-1};
%             OutData{iLayer}(cnn.Layers{iLayer-1}.OutDim+1:cnn.Layers{iLayer}.OutDim, :)=OptData;
        case 10
            % BLOB Layer
            if cnn.to.useGPU==1
                cnn.OutData{iLayer}=single(gpuArray.zeros(cnn.Layers{iLayer}.OutDim, numImages));
            else
                cnn.OutData{iLayer}=single(zeros(cnn.Layers{iLayer}.OutDim, numImages));
            end
%             offset=0;
            for inet=1:cnn.Layers{iLayer}.NNum
                tcnn=cnn.Layers{iLayer}.Nets{inet};
                tcnn=cnnFeedForward(tcnn, cnn.OutData{iLayer-1});
%                 cnn.OutData{iLayer}(offset+1:offset+tcnn.Layers{tcnn.LNum}.OutDim, :)=tcnn.OutData{tcnn.LNum};
%                 offset=offset+tcnn.Layers{tcnn.LNum}.OutDim;
                cnn.OutData{iLayer}=cnn.OutData{iLayer}+tcnn.OutData{tcnn.LNum}*norm(tcnn.OutData{tcnn.LNum});
                cnn.Layers{iLayer}.Nets{inet}=tcnn;
            end
        case 101
            % CS
%             if cnn.Layers{iLayer}.calced==0
                cnn.OutData{iLayer}=cnnCS(cnn.OutData{iLayer-1}, cnn.Layers{iLayer});
%                 cnn.Layers{iLayer}.calced=1;
%             end
        case 102
            % CoPCA
%             if cnn.Layers{iLayer}.calced==0
                cnn.OutData{iLayer}=cnnCoPCA(cnn.OutData{iLayer-1}, cnn.Layers{iLayer});
%                 cnn.Layers{iLayer}.calced=1;
%             end
        case 103
            % End
            cnn.OutData{iLayer}=cnn.OutData{iLayer-1};
        case 104
            % Transform
            cnn.OutData{iLayer}=cnnTransform(cnn.OutData{iLayer-1}, cnn.Layers{iLayer});
    end
end