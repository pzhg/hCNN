function acc=cnnTestData(cnn, VX, VY, numImages, to)
% Validate CNN Accuracy
%   VData: validation data, [x-dim, y-dim, channel-num, data-count]
%   VLabel: validation label, [1, data-count]
%   numImages: number of images that want to validate

images=VX(:, :, :, 1:numImages);
clear VData;
mb_labels=VY(:, 1:numImages);
clear VLabel;

% if to.PCAflag==1
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

cnn=cnnFeedForward(cnn, images);
[~, preds]=max(cnn.OutData{cnn.LNum}, [], 1);
acc=sum(preds==mb_labels)/numImages;