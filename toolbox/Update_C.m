function C_Mat = Update_C(D_Mat,P_Mat, X, A, par)

alpha = par.alpha;
Y_S = par.label;
C_Mat = (D_Mat'*D_Mat + alpha * P_Mat*A*A'*P_Mat'+ eye(size(D_Mat,2)))\(D_Mat'*X + alpha*P_Mat*A*Y_S');
end