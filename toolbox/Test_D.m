function [meanAcc,D_Mat,P_Mat] = Test_D(D_Mat, P_Mat, data,par)

img_fea = data.Xte;
semantic_uni = data.Yte;


cla_num = data.cla_num;
index = data.index;
sim_mat = img_fea' * D_Mat * P_Mat * semantic_uni;

rate = par.rate;

N = numel(index);            % total number
M = numel(unique(index));    % class number

confusion = zeros(M, 1);           % confusion vector
top1 = zeros(N, 1);
for i = 1 : N
    vec = sim_mat(i,:);
    [~, indx] = sort(vec, 'descend');
    top1(i) = indx(1);
end

begin = 1;
acc = zeros(M, 1);
for i = 1 : M
    confusion(i) = sum(top1(begin:begin+cla_num(i)-1, 1) == index(begin:begin+cla_num(i)-1, 1));   % all word
    begin = begin + cla_num(i);
    acc(i) = confusion(i) / cla_num(i);
end

meanAcc = mean(acc);

   X_t = [];
   A = [];
    num = zeros(M,1);
for j = 1 : M
    [loc,~] = find(top1 == j);
    tmp = zeros(size(loc,1),1);
    for k = 1:size(loc,1)
        tmp(k) = sim_mat(loc(k),j);
    end
    [local,val] = sort(tmp,'descend');
    num(j) = floor(size(loc,1) * rate);
    
    temp = loc(val);
    selected = temp(1:num(j));
    
    X_t =[X_t, img_fea(:,selected)];
    A = [A, repmat(semantic_uni(:,j),1,num(j))];
end

     X = X_t;
    [D_Mat, C_Mat] = Ini(X,A,D_Mat,P_Mat,par);
    
    % Alternatively updata D, P and C 
    iteration = 4;
    for i = 1:iteration
        D_Mat = Up_D(C_Mat, D_Mat, X, par);
        C_Mat = Up_C(D_Mat,P_Mat, X, A, par); 
    end
    meanAcc = TestDPL(D_Mat, P_Mat, data);
end


function [D, C] = Ini(X,A,D_Mat,P_Mat,par)

D = D_Mat;    % D
C = Up_C(D_Mat,P_Mat, X, A, par);
end

function C = Up_C(D_Mat,P_Mat, X, A, par)

lambda = par.lambda;

C = (D_Mat'*D_Mat + lambda*eye(size(D_Mat,2)))\(D_Mat'*X + lambda*P_Mat*A);
end

function D = Up_D(C_Mat, D_Mat, X, par)

mu = par.mu;

D = (X*C_Mat' + mu*D_Mat)/(C_Mat*C_Mat'+mu* eye(size(C_Mat,1)));
end

function meanAcc = accuracy(sim_mat, gt_label, testImCount)

N = numel(gt_label);            % total number
M = numel(unique(gt_label));    % class number

confusion = zeros(M, 1);           % confusion vector
top1 = zeros(N, 1);
for i = 1 : N
    vec = sim_mat(i,:);
    [~, indx] = sort(vec, 'descend');
    top1(i) = indx(1);
end

begin = 1;
acc = zeros(M, 1);
for i = 1 : M
    confusion(i) = sum(top1(begin:begin+testImCount(i)-1, 1) == gt_label(begin:begin+testImCount(i)-1, 1));   % all word
%     confusion(i) = sum(top1(begin:begin+testImCount(i)-1, 1) == i * ones(testImCount(i), 1));   % test word only
    begin = begin + testImCount(i);
    acc(i) = confusion(i) / testImCount(i);
end

meanAcc = mean(acc);
end
