function Delta_out=cnnDeActivate(ALayer, Delta_in, Data_2, Data_1)

switch ALayer.ActFunc
    case 'ReLu'
        % ReLu Layer
        Delta_out=Delta_in;
        Delta_out(real(Data_1)<0)=0;
        Delta_out(imag(Data_1)<0)=0;
    case 'LReLu'
        % Leaky ReLu Layer
        Delta_out=Delta_in;
        Delta_out(real(Data_1)<0)=ALayer.Leak*Delta_out(real(Data_1)<0);
        Delta_out(imag(Data_1)<0)=ALayer.Leak*Delta_out(imag(Data_1)<0);
    case 'ABS'
        % ABS Layer
        Re=Delta_in.*real(Data_1)./Data_2;
        Im=Delta_in.*imag(Data_1)./Data_2;
        Delta=Re+1j*Im;
    case 'Sigmoid'
        % Sigmoid Layer
        Delta_out=Delta_in.*Data_2.*(1-Data_2);
end