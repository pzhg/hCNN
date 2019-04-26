function fltOutput=cnnFilter(images, CLayer)

numImages=size(images, 4);
fltOutput=gpuArray.zeros(CLayer.FltDim*(CLayer.FltDim+1), numImages);


        
        for inum=1:numImages
            image=images(:, :, 1, inum);
            X=[];
%             X_cov=gpuArray.zeros(to.FDim(1)*to.FDim(2), to.FDim(1)*to.FDim(2));
            for ix=1:CLayer.PCAStep(1):size(images, 1)-CLayer.FDim(1)+1
                for iy=1:CLayer.PCAStep(2):size(images, 2)-CLayer.FDim(2)+1
                    temp=image(ix:ix+CLayer.FDim(1)-1, iy:iy+CLayer.FDim(2)-1);
                    temp=temp(:)-mean(temp(:));
                    X=[X, temp];
                end
            end
            X_cov=X'*X;
            [U, S, V]=svds(X_cov, CLayer.PCADim);
            PCAImage=U*S*V';
            PCAImage=PCAImage(:);
            CSImage=CLayer.A*image(:);
            fltOutput(1:CLayer.FltDim*CLayer.FltDim, inum)=PCAImage;
            fltOutput(CLayer.FltDim*CLayer.FltDim+1:CLayer.FltDim*(CLayer.FltDim+1), inum)=CSImage;
        end