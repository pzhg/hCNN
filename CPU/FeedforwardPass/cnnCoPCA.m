function fltOutput = cnnCoPCA(images, CLayer)

    numImages = size(images, 4);
    numFilter = size(images, 3);
    fltOutput = zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages);

    switch CLayer.CorrType
        case 1
            % Auto correlation
            parfor inum = 1:numImages
                for iflt = 1:numFilter
                    image = images(:, :, iflt, inum);
                    x_num = size(images, 1) - CLayer.FDim(1) + 1;
                    y_num = size(images, 2) - CLayer.FDim(2) + 1;
                    X = zeros(CLayer.FDim(1) * CLayer.FDim(2), floor((y_num - 1) / CLayer.PCAStep(2)) + 1);

                    for ix = 1:CLayer.PCAStep(1):x_num
                        y_index = 1;
                        for iy = 1:CLayer.PCAStep(2):y_num
                            temp = image(ix:ix + CLayer.FDim(1) - 1, iy:iy + CLayer.FDim(2) - 1);
                            temp = temp(:) - mean(temp(:));
                            X(:, y_index) = temp;
                            y_index = y_index + 1;
                        end
                    end
                    %             X=gpuArray(single(X));
                    X_cov = X' * X;
                    [U, S, V] = svds(X_cov, CLayer.PCADim);
                    PCAImage = U * S * V';
                    fltOutput(:, :, iflt, inum) = PCAImage;
                end
            end

        case 2
            % Cross correlation
            for inum = 1:numImages
                ofil = 1;

                for iflt1 = 1:numFilter
                    for iflt2 = iflt1 + 1:numFilter
                        image = images(:, :, iflt1, inum);
                        x_num = size(images, 1) - CLayer.FDim(1) + 1;
                        y_num = size(images, 2) - CLayer.FDim(2) + 1;
                        X = zeros(CLayer.FDim(1) * CLayer.FDim(2), floor((y_num - 1) / CLayer.PCAStep(2)) + 1);

                        for ix = 1:CLayer.PCAStep(1):x_num
                            y_index = 1;
                            for iy = 1:CLayer.PCAStep(2):y_num
                                temp = image(ix:ix + CLayer.FDim(1) - 1, iy:iy + CLayer.FDim(2) - 1);
                                temp = temp(:) - mean(temp(:));
                                X(:, y_index) = temp;
                                y_index = y_index + 1;
                            end
                        end

                        image = images(:, :, iflt2, inum);
                        x_num = size(images, 1) - CLayer.FDim(1) + 1;
                        y_num = size(images, 2) - CLayer.FDim(2) + 1;
                        Y = zeros(CLayer.FDim(1) * CLayer.FDim(2), floor((y_num - 1) / CLayer.PCAStep(2)) + 1);

                        for ix = 1:CLayer.PCAStep(1):x_num
                            y_index = 1;
                            for iy = 1:CLayer.PCAStep(2):y_num
                                temp = image(ix:ix + CLayer.FDim(1) - 1, iy:iy + CLayer.FDim(2) - 1);
                                temp = temp(:) - mean(temp(:));
                                Y(:, y_index) = temp;
                                y_index = y_index + 1;
                            end
                        end

                        X_cov = X' * Y;
                        [U, S, V] = svds(X_cov, CLayer.PCADim);
                        PCAImage = U * S * V';
                        fltOutput(:, :, ofil, inum) = PCAImage;
                        ofil = ofil + 1;
                    end
                end

            end

        otherwise
            error('Unknown Correlation Mode!');
    end
