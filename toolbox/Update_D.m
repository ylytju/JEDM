function D_Mat = Update_D(C_Mat, D_Mat, X)


Temp_S = D_Mat;
Temp_T = zeros(size(Temp_S));

rho = 1;
rate_rho = 0.999;
I_Mat = eye(size(C_Mat,1),size(C_Mat,1));

previous_D = D_Mat;
Iter = 1; Error = 1;
CC_Mat = C_Mat*C_Mat';
while(Error>1e-12&&Iter<50)
    Temp_D = (rho*(Temp_S-Temp_T)+X*C_Mat')/(rho*I_Mat+CC_Mat);
    Temp_S = normcol_lessequal(Temp_D+Temp_T);
    Temp_T = Temp_T+Temp_D-Temp_S;
    rho = rate_rho * rho;
    Error = mean(mean(previous_D-Temp_D).^2);
    previous_D = Temp_D;
    Iter = Iter + 1;
end
D_Mat = Temp_D;
end