addpath('../misc/','../tensorlab_2016-03-28/','./toolbox/ompbox/')

rng(10)
N = 64;     % Data dimension
K = 2000;   % Number of training samples
M = N*4;    % Number of atoms

% Data to approximate
Y = randn(N,K); % random data
Y = Y./repmat(sqrt(sum(Y.^2,1)),size(Y,1),1); % normalize data columns (optional)

% Initial dictionary
D = randn(N,M); D = D./repmat(sqrt(sum(D.^2,1)),size(D,1),1); % random
%D = odctndict([sqrt(N) sqrt(N)],M,2); % 2D-ODCT

%% Parameters for SuKro
params.iternum = 20;        % Number of iterations
params.memusage = 'high';   % Memory usage
params.initdict = D;        % Initial dictionary
params.data = Y;            % Data to approximate
params.alpha = 5;         % Number of separable terms
% Sparse coding parameters
params.codemode = 'sparsity';   % 'error' or 'sparsity'
params.Edata = 1e-1;            % when using codemode = 'error'
params.Tdata = ceil(size(D,1)/10);     % when using codemode = 'sparsity'
% Dimensions of the subdictionaries
N1 = sqrt(N); N2 = sqrt(N);
M1 = sqrt(M); M2 = sqrt(M);
params.kro_dims.N = [N1 N2];
params.kro_dims.M = [M1 M2];

assert(length(params.kro_dims.N) == length(params.kro_dims.M), ...
        'N and M vectors should have same size')

%% Running SuKro algorithm
flops_dense = 2*size(Y,1)*size(Y,2);

[D, D_not_normalized, X] = HO_SuKro_DL(params);
  
% Verifying the number of separable terms on the learned dictionary
D_reord = rearrangement_recursive(D_not_normalized,params.kro_dims.N,params.kro_dims.M);
fprintf('Learned dictionary has %d separable terms.\n',rank(D_reord,norm(D_reord,'fro')*2e-7));
norm(Y-D*X,'fro')/norm(Y,'fro')