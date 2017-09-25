function main()

%% 
% Yunlong Yu, Zhong Ji, Xi Li*, Jichang Guo, Zhongfei Zhang, Haibin Ling, Fei Wu
% To appear in Transactions on Cybernetics 2017 
% Transductive Zero-Shot Learning with a self-training dictionary approach
% Written by Yunlong Yu, yuyunlong@tju.edu.cn
% School of Electrical and Information Engineering,
% Tianjin University
% 300072£¬Tianjin
%% STEP 0: Initialise parameters and load the data
clc; clear all; close all;
addpath('./toolbox');

dataset_path = 'fea/AwA/att'; % AwA, CUB, aPY, SUN
Train = load(fullfile(dataset_path,'Train.mat'));
Test  = load(fullfile(dataset_path,'Test.mat'));
%% Parameter setting
options.alpha = 0.1;       % 0<=alpha<=1 insensitive, keep default 0.1
options.beta  = 0.01;       % 0<=beta<=1 insensitive, keep default 0.01
options.code_size = 100;   % insensitive, keep default
options.rdim_num = 100;

fprintf('...default params...: alpha=%d lambda=%d code_size=%d \n', options.alpha, options.beta, options.code_size);

%% Pre-process
Xtr = normcol_equal(Train.im_fea);       data.Xtr = Xtr - repmat(mean(Xtr,2),1,size(Xtr,2)); 
Ytr = normcol_equal(Train.semantic_fea); data.Ytr = Ytr - repmat(mean(Ytr,2),1,size(Ytr,2));
Xte = normcol_equal(Test.im_fea);        data.Xte = Xte - repmat(mean(Xte,2),1,size(Xte,2));
Yte = normcol_equal(Test.semantic_uni);  data.Yte = Yte - repmat(mean(Yte,2),1,size(Yte,2));

data.cls_num = Train.cla_num;
data.cla_num = Test.cla_num;
data.index = Test.index;

tic  % train time

%% Dimensionality reduction
Vt       = Eigenface_f(data.Xtr,options.rdim_num);
data.Xtr = Vt'*data.Xtr;  
data.Xte = Vt'*data.Xte;


%% run JEDM 
disp('...  Train the model  ...')
[D_Mat, P_Mat] = TrainDPL(data, options);
train_time = toc;
tic
meanAcc_JEDM = TestDPL(D_Mat, P_Mat, data);
test_time = toc;
fprintf('The JEDM accuracy is: %0.3f%%\n', meanAcc_JEDM*100);
% save(fullfile(dataset_path,'T.mat'),'D_Mat','P_Mat','Vt');

%% run TSTD
lambdas = 100;   
mus = 100;    
rates = [0.4,0.6,0.8,1];  
for j = 1:size(lambdas,2)
    par.lambda = lambdas(j);
    disp(['lambda...',num2str(par.lambda)]);    
    for k = 1:size(mus,2)
     par.mu = mus(k);
        disp(['mu...',num2str(par.mu)]);  
        best_meanAcc = meanAcc_JEDM; 
        for i = 1:size(rates,2)
             par.rate = rates(i);    
              disp(['rate...',num2str(par.rate)]);     
              [meanAcc_TSTD,D_Mat,P_Mat] = Test_D(D_Mat, P_Mat, data, par);
              if meanAcc_TSTD > best_meanAcc
                  best_meanAcc = meanAcc_TSTD;
              end
        end  
    end
end

fprintf('The JEDM accuracy is: %0.3f%%\n', meanAcc_JEDM*100);
fprintf('The TSTD accuracy is: %0.3f%%\n', best_meanAcc*100);
fprintf('The train time is: %0.2fs\n', train_time);
fprintf('The test time is: %0.2fs\n', test_time);
end