function [D_Mat,P_Mat, C_Mat] = Initilization (X,A,par)

code_size = par.code_size;
Dim_x = size(X,1);   % 4096
Dim_y = size(A,1);   % 85

D_Mat = normcol_equal(randn(Dim_x,code_size));    % D
P_Mat = normcol_equal(randn(code_size, Dim_y));     % P

C_Mat = Update_C(D_Mat,P_Mat, X, A, par);

end