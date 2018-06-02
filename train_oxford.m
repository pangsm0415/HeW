clear

addpath .\utils
load('.\data\oxford5k_cnn.mat')

%% For centering the deep features 
nfeats = 0;
mean_CNN = zeros(512,1,'single');

for i = 1:size(Oxford5k_cnn,2)
    CNN = Oxford5k_cnn{i};
    CNN = reshape(CNN,[],512);
    CNN = CNN'; 
    mean_CNN = mean_CNN + sum(CNN,2);
    nfeats = nfeats + size(CNN,2);
end
mean_CNN = mean_CNN/nfeats;

save('mean_CNN', 'mean_CNN')

%% computing the representations of training set
vecs_ox = cell(1,length(Oxford5k_cnn));
for i = 1:length(Oxford5k_cnn)
    CNN = Oxford5k_cnn{i};
    Oxford5k_cnn{i} = [];
    [CNN_org, Fweights] = Weight_Heat(CNN, mean_CNN);

    CNN = bsxfun(@times,CNN_org, 1./Fweights);
    vecs_ox{i} = sum(CNN,2);  
end
save('vecs_ox','vecs_ox');

%% main function
function [CNN_org, Fweights] = Weight_Heat(CNN, mean_CNN)
[W,H,K] = size(CNN);
S = sum(CNN,3);

CNN = reshape(CNN,[],K);
CNN = CNN'; 
CNN_org = CNN; 
CNN = CNN - mean_CNN;
CNN = yael_vecs_normalize(CNN,2,0);

S0 = reshape(S,[],1);
A = CNN'*CNN;
ind = find(S0==0);
A (1:size(A,1)+1:size(A,1)^2) = 0;
A(ind,ind) = 0;
A(A<0) = 0;

constZ = 0.1;
Z =  constZ*mean(A(A>0)) ; % conductance to dummy ground

%% Weights
Fweights = get_potential_inv(A, Z);%ÓÐweight
end

function reward = get_potential_inv(A, Z)

A = [A, Z*ones(size(A,1),1,'single');  ones(1,size(A,1),'single'), 0];
sA = sum(A,2);
A  = bsxfun(@rdivide, A, sA);

% Laplace
lap_mat = diag(sum(A,2)) - A ;
lap_mat = lap_mat(1:end-1,1:end-1);

% compute inverse of Laplace
inv_lap_mat = inv(lap_mat) ;

%% compute weights in practice:
deno = inv_lap_mat(1:size(lap_mat,1)+1:size(lap_mat,1)^2);
inv_lap_mat(1:size(inv_lap_mat,1)+1:size(inv_lap_mat,1)^2) = 0;
reward = sum(inv_lap_mat)./deno;
reward = (reward)/size(A,1);

%% compute weights in practice:
% lap_mat(1:size(lap_mat,1)+1:size(lap_mat,1)^2) = 0;
% lap_mat = -lap_mat;
% inv_lap_mat(1:size(inv_lap_mat,1)+1:size(inv_lap_mat,1)^2) = 0;
% deno = sum(lap_mat'.*inv_lap_mat);
% pot = bsxfun(@rdivide, inv_lap_mat, 1+deno);
% reward = sum(pot);
end
