function fltOutput=cnnCoPCA(images, CLayer)

numImages=size(images, 4);
numFilter=size(images, 3);
fltOutput=single(gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages));
        
if CLayer.CorrType==1 
    % Auto correlation
    for inum=1:numImages
        for iflt=1:numFilter
            image=images(:, :, iflt, inum);
            X=gpuArray(single([]));
            for ix=1:CLayer.PCAStep(1):size(images, 1)-CLayer.FDim(1)+1
                for iy=1:CLayer.PCAStep(2):size(images, 2)-CLayer.FDim(2)+1
                    temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                    temp=temp(:)-mean(temp(:));
                    X=[X, temp];
                end
            end
%             X=gpuArray(single(X));
            X_cov=X'*X;
            [U, S, V]=svds(double(X_cov), CLayer.PCADim);
            PCAImage=U*S*V';
            fltOutput(:, :, iflt, inum)=single(PCAImage);
        end
    end
else
    % Cross correlation
    for inum=1:numImages
        ofil=1;
        for iflt1=1:numFilter
            for iflt2=iflt1+1:numFilter
                image=images(:, :, iflt1, inum);
                X=gpuArray(single([]));
                for ix=1:CLayer.PCAStep(1):size(images, 1)-CLayer.FDim(1)+1
                    for iy=1:CLayer.PCAStep(2):size(images, 2)-CLayer.FDim(2)+1
                        temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                        temp=temp(:)-mean(temp(:));
                        X=[X, temp];
                    end
                end
                image=images(:, :, iflt2, inum);
                Y=gpuArray(single([]));
                for ix=1:CLayer.PCAStep(1):size(images, 1)-CLayer.FDim(1)+1
                    for iy=1:CLayer.PCAStep(2):size(images, 2)-CLayer.FDim(2)+1
                        temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                        temp=temp(:)-mean(temp(:));
                        Y=[Y, temp];
                    end
                end
                X_cov=X'*Y;
                [U, S, V]=svds(double(X_cov), CLayer.PCADim);
                PCAImage=U*S*V';
                fltOutput(:, :, ofil, inum)=single(PCAImage);
                ofil=ofil+1;
            end
        end
    end
end