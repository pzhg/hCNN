function Delta_out = cnnDeActivate(ALayer, Delta_in, Data_2, Data_1)

    switch ALayer.ActFunc
        case {'ReLu', 'reLu', 'Relu', 'relu'}
            % ReLu Layer
            Delta_out = Delta_in;
            Delta_out(real(Data_1) < 0) = 0;
            Delta_out(imag(Data_1) < 0) = 0;
        case {'LReLu', 'LreLu', 'LRelu', 'lrelu'}
            % Leaky ReLu Layer
            Delta_out = Delta_in;
            Delta_out(real(Data_1) < 0) = ALayer.Leak * Delta_out(real(Data_1) < 0);
            Delta_out(imag(Data_1) < 0) = ALayer.Leak * Delta_out(imag(Data_1) < 0);
        case {'ABS', 'abs'}
            % ABS Layer
            Re = Delta_in .* real(Data_1) ./ Data_2;
            Im = Delta_in .* imag(Data_1) ./ Data_2;
            Delta_out = Re + 1j * Im;
        case {'Sigmoid', 'sigmoid'}
            % Sigmoid Layer
            Delta_out = Delta_in .* Data_2 .* (1 - Data_2);
        otherwise
            error('Unknown Activation Function Type!');
    end
    
end