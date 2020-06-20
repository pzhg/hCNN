function fltOutput=cnnCoPCA_GPU(images, CLayer)

numImages=size(images, 4);
numFilter=size(images, 3);
fltOutput=gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages, 'single');
        
if CLayer.CorrType==1 
    % Auto correlation
    parfor inum=1:numImages
        for iflt=1:numFilter
            image=images(:, :, iflt, inum);
            x_num=size(images, 1)-CLayer.FDim(1)+1;
            y_num=size(images, 2)-CLayer.FDim(2)+1;
            X=gpuArray.zeros(CLayer.FDim(1)*CLayer.FDim(2), floor((y_num-1)/CLayer.PCAStep(2))+1, 'single');
            for ix=1:CLayer.PCAStep(1):x_num
                y_index=1;
                for iy=1:CLayer.PCAStep(2):y_num
                    temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                    temp=temp(:)-mean(temp(:));
                    X(:, y_index)=temp;
                    y_index=y_index+1;
                end
            end
%             X=gpuArray(single(X));
            X_cov=X'*X;
            [U, S, V]=svds(gather(X_cov), CLayer.PCADim);
            PCAImage=U*S*V';
            fltOutput(:, :, iflt, inum)=gpuArray(single(PCAImage));
        end
    end
else
    % Cross correlation
    for inum=1:numImages
        ofil=1;
        for iflt1=1:numFilter
            for iflt2=iflt1+1:numFilter
                image=images(:, :, iflt1, inum);
                x_num=size(images, 1)-CLayer.FDim(1)+1;
                y_num=size(images, 2)-CLayer.FDim(2)+1;
                X=gpuArray.zeros(CLayer.FDim(1)*CLayer.FDim(2), floor((y_num-1)/CLayer.PCAStep(2))+1, 'single');
                for ix=1:CLayer.PCAStep(1):x_num
                    y_index=1;
                    for iy=1:CLayer.PCAStep(2):y_num
                        temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                        temp=temp(:)-mean(temp(:));
                        X(:, y_index)=temp;
                        y_index=y_index+1;
                    end
                end
                image=images(:, :, iflt2, inum);
                x_num=size(images, 1)-CLayer.FDim(1)+1;
                y_num=size(images, 2)-CLayer.FDim(2)+1;
                Y=gpuArray.zeros(CLayer.FDim(1)*CLayer.FDim(2), floor((y_num-1)/CLayer.PCAStep(2))+1, 'single');
                for ix=1:CLayer.PCAStep(1):x_num
                    y_index=1;
                    for iy=1:CLayer.PCAStep(2):y_num
                        temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                        temp=temp(:)-mean(temp(:));
                        Y(:, y_index)=temp;
                        y_index=y_index+1;
                    end
                end
                X_cov=X'*Y;
                [U, S, V]=svds(gather(X_cov), CLayer.PCADim);
                PCAImage=U*S*V';
                fltOutput(:, :, ofil, inum)=gpuArray(single(PCAImage));
                ofil=ofil+1;
            end
        end
    end
end