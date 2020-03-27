% Dictionary learning experiment with synthetic data following the same
% setup as in:
% [1] Ghassemi et al. 2017, 'STARK: Structured Dictionary Learning 
% Through Rank-one Tensor Recovery'.
%
% A Kronecker-structured dictionary is built, from which the training data
% is generated as Y = DX, with X a sparse matrix with fixed number of
% nonzero gaussian entries (N(0,1)) by column in a uniformly distributed
% support.

% However, in [1] the authors do not specify if noise is added. They don't
% specify either if the sparsity of the learned X (informed to OMP, 
% supposing that OMP is used on the sparsity mode instead of noise mode) 
% is the same as the ground truth. They don't specify how the approxmation
% error is calculated.
%
% CONCLUSIONS SO FAR: could not repoduce results. I obtain a different
% order of magnitude on the approximation error.

addpath('../misc/','../tensorlab_2016-03-28/','./toolbox/ompbox/','./toolbox/ksvdbox/')

rng(100)

%% Parameters
% Dimensions of the subdictionaries
% N1 = 4;     N2 = 6;
% M1 = 12;    M2 = 8;

N1 = 2;     N2 = 5;     N3 = 5;
M1 = 4;     M2 = 10;    M3 = 5;

N1 = 2;     N2 = 5;     N3 = 10;
M1 = 4;     M2 = 10;    M3 = 20;

N_vec = [N1 N2 N3];
M_vec = [M1 M2 M3];

N = prod(N_vec);    % Data dimension
M = prod(M_vec);    % Number of atoms
K = 5000;  %[500 1000 2500 5000] % Number of training samples

alpha = 3;  % Number of separable terms

% Sparsity of columns of X
s = 10;

%% Data generation
% Dictionary generation  D = \sum{kron(A_i,B_i,...,Zi)}
D = zeros(N,M);
D_terms = cell(length(N_vec),alpha); % Stores all submatrices
for r = 1:alpha
    D_terms{1,r} = randn(N_vec(1),M_vec(1)); 
    D_terms{1,r} = D_terms{1,r}./repmat(sqrt(sum(D_terms{1,r}.^2,1)),size(D_terms{1,r},1),1); % normalize columns
    D_alpha = D_terms{1,r};
    for k = 2:length(N_vec) % go over each submatrix and take the kroneker product
        D_terms{k,r} = randn(N_vec(k),M_vec(k));
        D_terms{k,r} = D_terms{k,r}./repmat(sqrt(sum(D_terms{k,r}.^2,1)),size(D_terms{k,r},1),1); % normalize columns
        D_alpha = kron(D_alpha,D_terms{k,r});
    end
    D = D + 1/r*D_alpha;
end
clear D_alpha

% alpha > 1 requires final normalization on D atoms
D_not_normalized = D;
D = D./repmat(sqrt(sum(D.^2,1)),size(D,1),1);

% Sparse coefficient matrix
% indices of nonzero entries (uniformily distributed)
supp_idx = zeros(s,K);
for k = 1:K
   supp_idx(:,k) = (k-1)*M + randperm(M,s).'; 
end
%supp_idx = ceil(M*rand(s,K)); % PS: some indices might duplicate, meaning that some columns will have less nonzero entries than demanded
%supp_idx = supp_idx + repmat((0:K-1)*M,s,1);
X = zeros(M,K);
X(supp_idx(:)) = randn(1,s*K);

% Data to approximate
Y = D*X; % random data

% Initial dictionary
D_hat = Y(:,randperm(K,M)); D_hat = D_hat./repmat(sqrt(sum(D_hat.^2,1)),size(D_hat,1),1);       % Initial dictionary

%% Parameters for SuKro
params.iternum = 20;        % Number of iterations
params.memusage = 'high';   % Memory usage
params.initdict = D_hat;    % Initial dictionary
params.data = Y;            % Data to approximate
params.alpha = alpha;         % Number of separable terms
% Sparse coding parameters
params.codemode = 'sparsity';   % 'error' or 'sparsity'
params.Edata = 1e-1;            % when using codemode = 'error'
params.Tdata = 2*s;     % when using codemode = 'sparsity'
% Dimensions of the subdictionaries
params.kro_dims.N = N_vec;
params.kro_dims.M = M_vec;

assert(length(params.kro_dims.N) == length(params.kro_dims.M), ...
        'N and M vectors should have same size')

%% Running SuKro algorithm
profile on
[D_hat, D_not_normalized_hat, X_hat] = HO_SuKro_DL(params);
profile off
profsave(profile('info'), strcat('myprofile_DL_synth_data'))

% KSVD
[D_hat_KSVD,X_hat_KSVD,~,~] = ksvd(params);


% Verifying the number of separable terms on the learned dictionary
D_reord = rearrangement_recursive(D_not_normalized_hat,params.kro_dims.N,params.kro_dims.M);
fprintf('Learned dictionary has %s separable terms.\n',mlrank(D_reord));

%% Approximation error
norm(Y-D_hat*X_hat,'fro')/norm(Y,'fro')
norm(Y-D_hat_KSVD*X_hat_KSVD,'fro')/norm(Y,'fro')

% RMSE as in KSVD
sqrt(sum(sum((Y-D_hat*X_hat).^2))/(N*K))
sqrt(sum(sum((Y-D_hat_KSVD*X_hat_KSVD).^2))/(N*K))