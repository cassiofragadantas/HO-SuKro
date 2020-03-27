% Same as test.m but with a for loop on different  input SNR and MC iterations.

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

SNR_vec = -10:5:30;
SNR_fout = zeros(size(SNR_vec));
mc_it = 10;

for k_mc = 1:mc_it
%% Constructing D = \sum{kron(A_i,B_i)}
D = zeros(prod(n),prod(m));
D_terms = cell(length(n),alpha); % Stores all submatrices
for r = 1:alpha
    D_terms{1,r} = randn(n(1),m(1));
    D_terms{1,r} = normc(D_terms{1,r});
    D_alpha = D_terms{1,r};
    for k = 2:length(n) % go over each submatrix and take the kroneker product
        D_terms{k,r} = randn(n(k),m(k));
        D_terms{k,r} = normc(D_terms{k,r});
        D_alpha = kron(D_alpha,D_terms{k,r});
    end
    D = D + 1/r*D_alpha;
end
clear D_alpha

for k_SNR = 1:length(SNR_vec);
%% Adding Noise
SNR_db = SNR_vec(k_SNR);
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

%% Alternative reconstruction
Dhat2 = zeros(prod(n),prod(m));
D_terms_hat = cell(length(n),alpha); % Stores all submatrices
for r = 1:alpha
    D_terms_hat{end,r} = reshape(Uhat{1}(:,r),n(end),m(end));
    D_terms_hat{end,r} = normc(D_terms_hat{end,r});
    Dsep = D_terms_hat{end,r};
    for k=2:length(n)
        D_terms_hat{end-k+1,r} = reshape(Uhat{k}(:,r),n(end-k+1),m(end-k+1));
        Dsep = kron(D_terms_hat{end-k+1,r}, Dsep);
    end
    Dhat2 = Dhat2 + Dsep; 
end

%% SNR
SNR_out = 10*log10(norm(D,'fro')/(norm(D-Dhat,'fro'))^2);
% by factors
% 2 issues to consider: constant multiplication factor ambiguity and
% ordering of recovered factors (on the dimension of r)
signal = 0;
noise = 0;
noise_r = zeros(1,alpha);
for r = 1:alpha
    for k = 1:k
        signal = signal + norm(normc(D_terms{k,r}),'fro')^2;
        for rhat = 1:alpha % dumb way to treate ordering ambiguity on recovery
            noise_r(rhat) =  min( norm(normc(D_terms_hat{k,rhat})-normc(D_terms{k,r}),'fro')^2, ...
                                  norm(normc(D_terms_hat{k,rhat})+normc(D_terms{k,r}),'fro')^2);
        end
        noise = noise + min(noise_r);
    end
end
SNR_fout(k_SNR) = SNR_fout(k_SNR) + 10*log10(signal/noise);

end
k_mc
end
SNR_fout = SNR_fout/mc_it

plot(SNR_vec, SNR_fout)
hold on, plot(SNR_vec, SNR_vec,':')
xlabel('SNR_{in} [dB]'), ylabel('SNR_{f-out} [dB]')
legend('Recovered','Baseline')

%% Verifying the result
% fprintf('\n Rank informed to CPD is the same as the number of separable terms: %d',alpha)
% fprintf('\n Elapsed time on CPD calculation: %.2f s',elapsed_time)
% error_signal_ratio = (norm(D-Dhat,'fro')/norm(D,'fro'))^2;
% fprintf('\n Approximation error ||D - Dhat||/||D|| (Frobenius norm): %f\n',error_signal_ratio)
% 
% fprintf('\n Input SNR (dB): %.1f',SNR_db)
% fprintf('\n Approximation error SNR (dB) w.r.t. oracle: %f\n',10*log10(1/error_signal_ratio))

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