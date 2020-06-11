
function actOutput=cnnActivate(ALayer, actInput)

numImages=size(actInput, 4);

switch ALayer.ActFunc
    case 'ReLu'
        % ReLu Layer
        actOutput=actInput;
        actOutput(real(actInput)<0)=0;
        actOutput(imag(actInput)<0)=0;
    case 'LReLu'
        % Leaky ReLu Layer
        actOutput=actInput;
        actOutput(real(actInput)<0)=ALayer.Leak*actInput(real(actInput)<0);
        actOutput(imag(actInput)<0)=ALayer.Leak*actInput(imag(actInput)<0);
    case 'ABS'
        % ABS Layer
        absImage=abs(actInput);
        maxImage=max(max(absImage(:, :, 1, 1:numImages)));
        actOutput=absImage./maxImage;
    case 'Sigmoid'
        % Sigmoid Layer
        actOutput=1./(1+exp(-actInput));
end