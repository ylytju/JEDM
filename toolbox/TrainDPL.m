function [D_Mat, P_Mat] = TrainDPL(data,params)

 X = data.Xtr;
 Y = data.Ytr;
 par.code_size = params.code_size;
 par.alpha = params.alpha;
 par.beta  = params.beta;
 N = length(data.cls_num);
 img_label = [];
 for i = 1:N
    img_label = [img_label;ones(data.cls_num(i),1)*i]; 
 end     
 for i = 1:sum(data.cls_num)
    label(i,img_label(i)) = 1; 
 end
 
par.label = label;
A = unique(Y','rows','stable')';
[D_Mat,P_Mat, C_Mat] = Initilization(X,A,par);

%% Alternatively updata D, P and C 
iteration = 20;
for i = 1:iteration
%    fprintf('...iteration...%d\n',i);
   P_Mat = Update_P(C_Mat,A,par);
   D_Mat = Update_D(C_Mat, D_Mat, X);  
   C_Mat = Update_C(D_Mat,P_Mat, X, A, par); 
end
end