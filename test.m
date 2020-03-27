addpath('./tensorlab_2016-03-28/','./misc/')
%% User-defined Parameters
% Dimensions of sub-matrices: A (n1xm1), B (n2xm2) ...
% n = [n1 n2 ... nK] where K is the tensor order
n = [5 4 3];
% m = [m1 m2 ... mK] where K is the tensor order
m = [6 5 4];

assert(length(n) == length(m)) % n and m should have same size

% Number of separable terms (alpha)
alpha = 3;

%% Constructing D = \sum{kron(A_i,B_i)}
D = zeros(prod(n),prod(m));
D_terms = cell(length(n),alpha); % Stores all submatrices
for r = 1:alpha
    D_terms{1,r} = randn(n(1),m(1));
    D_alpha = D_terms{1,r};
    for k = 2:length(n) % go over each submatrix and take the kroneker product
        D_terms{k,r} = randn(n(k),m(k));
        D_alpha = kron(D_alpha,D_terms{k,r});
    end
    D = D + 1/r*D_alpha;
end
clear D_alpha

%% Adding Noise
SNR_db = 20;
N = randn(size(D)); N = N/norm(N,'fro')*norm(D,'fro'); % Noise with same power as D
D_noisy = D + 10^(-SNR_db/20)*N;

%% Rearrangement
R_D = rearrangement_recursive(D_noisy,n,m);

%% CPD via tensorlab
tic;
Uhat = cpd(R_D,alpha); % supposing underlying rank (alpha) known
elapsed_time = toc;
R_Dhat = cpdgen(Uhat);

%% Inverse rearrangement
Dhat = rearrangement_inv_recursive(R_Dhat,n,m);

%% Verifying the result
fprintf('\n Rank informed to CPD is the same as the number of separable terms: %d',alpha)
fprintf('\n Elapsed time on CPD calculation: %.2f s',elapsed_time)
error_signal_ratio = (norm(D-Dhat,'fro')/norm(D,'fro'))^2;
fprintf('\n Approximation error ||D - Dhat||/||D|| (Frobenius norm): %f\n',error_signal_ratio)

fprintf('\n Input SNR (dB): %.1f',SNR_db)
fprintf('\n Approximation error SNR (dB) w.r.t. oracle: %f\n',10*log10(1/error_signal_ratio))

%% Visualizing
show_plots = false;
if show_plots
    visualization
end

%% RC
calculate_RC = false;
if calculate_RC
    RC
    fprintf('\n Empirical RC: %f\n',RC_empirical)
end

% Correlated factors
% if r ==1
%     D_terms{k,r} = randn(n(k),m(k));
% else
%     D_terms{k,r} = 1*D_terms{k,r-1} + 0*randn(n(k),m(k));
% end