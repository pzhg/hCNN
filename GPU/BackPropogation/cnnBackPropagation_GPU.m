function cnn=cnnBackPropagation_GPU(cnn, outPut)

cnn.W_grad=cell(1, cnn.LNum);
cnn.B_grad=cell(1, cnn.LNum);
for iLayer=cnn.LNum:-1:1
    switch cnn.Layers{iLayer}.type
        case 0
            % Input
            cnn.Delta{iLayer}=cnn.Delta{iLayer+1};
        case {4, 8}
            % Error of SoftMax Layer
            cnn.Delta{iLayer}=cnn.OutData{iLayer}-outPut;
        case 3
            % Error of Fully Connected Layer
            cnn.Delta{iLayer}=cnnDeFullConnected(cnn.Layers{iLayer}, cnn.Delta{iLayer+1});
        case 5
            % Error of Pooling Layer
            cnn.Delta{iLayer}=cnnDePool_GPU(cnn.Layers{iLayer}, cnn.Delta{iLayer+1});
        case 7
            % Error of Activation Layer
            cnn.Delta{iLayer}=cnnDeActivate(cnn.Layers{iLayer}, cnn.Delta{iLayer+1}, cnn.OutData{iLayer}, cnn.OutData{iLayer-1});
        case 2
            % Error of Convolution Layer
            cnn.Delta{iLayer}=cnnDeConv_GPU(cnn.Layers{iLayer}, cnn.Delta{iLayer+1});
        case 6
            % Error of Reshape Layer
%             if iLayer==cnn.LNum
%                 cnn.Delta{iLayer}=reshape(outPut, size(cnn.OutData{iLayer-1}));
%             else
                cnn.Delta{iLayer}=reshape(cnn.Delta{iLayer+1}, size(cnn.OutData{iLayer-1}));
%             end
        case 1
            % Error of Hybrid Convolution Layer
            [Ka, Kr]=cnnDeconvolveRadar_GPU(cnn.Layers{iLayer}, cnn.Delta{iLayer+1}, cnn.OutData{iLayer-1});
            cnn.Delta{iLayer}.Ka=Ka;
            cnn.Delta{iLayer}.Kr=Kr;
        case 9
            % (Deprecated) SP Filter Layer
            cnn.Delta{iLayer}=cnn.Delta{iLayer+1}(1:cnn.Layers{iLayer-1}.OutDim, :);
        case 10
            % BLOB Layer
%             offset=0;
            if cnn.Layers{iLayer}.combineType==1
                for inet=1:cnn.Layers{iLayer}.NNum
                    tcnn=cnn.Layers{iLayer}.Nets{inet};
%                 Delta=cnn.Delta{iLayer+1}(offset+1:offset+tcnn.Layers{tcnn.LNum}.OutDim, :);
%                 offset=offset+tcnn.Layers{tcnn.LNum}.OutDim;
%                 tcnn=cnnBackPropagation(tcnn, Delta);
                    tcnn=cnnBackPropagation_GPU(tcnn, cnn.Delta{iLayer+1});
                    if tcnn.Layers{1}.type<9
                        cnn.Delta{iLayer}=tcnn.Delta{1};
                    end
                    cnn.Layers{iLayer}.Nets{inet}=tcnn;
                end
            else
                offset=0;
                for inet=1:cnn.Layers{iLayer}.NNum
                    tcnn=cnn.Layers{iLayer}.Nets{inet};
                    Delta=cnn.Delta{iLayer+1}(offset+1:offset+tcnn.Layers{tcnn.LNum}.OutDim, :);
                    offset=offset+tcnn.Layers{tcnn.LNum}.OutDim;
                    tcnn=cnnBackPropagation_GPU(tcnn, Delta);
                    if tcnn.Layers{1}.type<9
                        cnn.Delta{iLayer}=tcnn.Delta{1};
                    end
                    cnn.Layers{iLayer}.Nets{inet}=tcnn;
                end
            end
            
        case 101
            % CS
        case 102
            % CoPCA
        case 103
            % End
            cnn.Delta{iLayer}=outPut;
        case 104
            % Transform
        case 11
            [cnn.Delta{iLayer}, dgamma, dbeta]=cnnDeBatchedNormalization(cnn.Layers{iLayer}, cnn.Delta{iLayer+1});
            cnn.W_grad{iLayer}.dgamma=dgamma;
            cnn.W_grad{iLayer}.dbeta=dbeta;
    end
end