clear
addpath .\utils
addpath .\data
%------------------------PCA training------------------------
fprintf('Learning PCA parameters...\n');
load('vecs_ox.mat')
vecs_train = cell2mat(vecs_ox);
clear vecs_ox

vecs_train = vecs_train.^0.5;
vecs_train = yael_vecs_normalize(vecs_train,2,0);
[~, eigvec, eigval, Xm] = yael_pca (vecs_train);

%---------------------Process test dataset------------------------
fprintf('Processing test dataset...\n');
load vecs_ho.mat
X = cell2mat(vecs_ho);
clear vecs_ho
X = X.^0.5;
X = yael_vecs_normalize(X,2,0);

X = apply_whiten (X, Xm, eigvec, eigval);
X = yael_vecs_normalize(X,2,0);
fprintf('end...\n');

%----------------------query process---------------------
fprintf('Processing query images...\n');
load gnd_holidays.mat
Q = X(:,qidx);
fprintf('end...\n');

%-------------------image search--------------------
[ranks,sim] = yael_nn(X, Q, size(X,2), 'L2');
[map,aps] = compute_map (ranks, gnd);
