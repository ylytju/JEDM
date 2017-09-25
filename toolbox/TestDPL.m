function [meanAcc] = TestDPL(D_Mat, P_Mat, data)


%
img_fea = data.Xte;   
semantic_uni = data.Yte;
cla_num = data.cla_num;
index = data.index;

W_Mat = normcol_equal(D_Mat * P_Mat);
sim = normcol_equal(img_fea' * W_Mat * semantic_uni);
meanAcc = accuracy(sim,index,cla_num);

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