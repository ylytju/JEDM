function P_Mat = Update_P(C_Mat,A,par)

% updata the Projection matrix
alpha = par.alpha;
beta = par.beta;
Y_S = par.label;
gamma = beta/alpha;
I_Mat = eye(size(C_Mat,1),size(C_Mat,1));
temp = C_Mat*Y_S*A'/(A*A'+eye(size(A,1)));
P_Mat = (C_Mat*C_Mat'+gamma*I_Mat)\temp;
end