if ~exist('cpd_rank','var')
    cpd_rank = size(D_terms,2); %size(Uhat{1},2);
end

%% Theoretical RC
% product D*x
if length(n) == 3
    cost_struct = min([n(1)*m(1)*m(2)*m(3) + n(2)*m(2)*n(1)*m(3) + n(3)*m(3)*n(1)*n(2), ...
                       n(1)*m(1)*m(2)*m(3) + n(3)*m(3)*n(1)*m(2) + n(2)*m(2)*n(1)*n(3), ...
                       n(2)*m(2)*m(1)*m(3) + n(1)*m(1)*n(2)*m(3) + n(3)*m(3)*n(1)*n(2), ...
                       n(2)*m(2)*m(1)*m(3) + n(3)*m(3)*n(2)*m(1) + n(2)*m(2)*n(2)*n(3), ...
                       n(3)*m(3)*m(1)*m(2) + n(1)*m(1)*n(3)*m(2) + n(2)*m(2)*n(3)*n(1), ...
                       n(3)*m(3)*m(1)*m(2) + n(2)*m(2)*n(3)*m(1) + n(1)*m(1)*n(3)*n(2)]);
    RC_theoretical = cpd_rank*cost_struct/(prod(n)*prod(m));
end

%% Practical RC

% product D*x
mc_it = 10000;
mean_time_dense = 0;
mean_time_sukro = 0;
mean_time_mode = zeros(size(n));


for k_mc = 1:mc_it
    
    x = randn(prod(m),1); % random vector
    X = reshape(x,m); % tensorize the vector
    
    
    % Dense p
    tic
    y = D*x;
    prod_time = toc;
    mean_time_dense = mean_time_dense + prod_time/mc_it;

    % ---- SUKRO TIME ----

%     tic
    X = reshape(x,m); % tensorize the vector
%     X = tensor(reshape(x,m));
    Y = zeros(n);
    tic
    for r = 1:cpd_rank
%         Y = Y + tmprod(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + modeprod(X,D_terms(:,r),1:length(m),m); % multiplication
        Y = Y + modeprod3(X,D_terms{1,r},D_terms{2,r},D_terms{3,r},m); % multiplication
%         Y = Y + ttm(X,D_terms(:,r),1:length(m)); % multiplication
    end
    y2 = Y(:); %vectorize the result
    prod_time = toc;
    mean_time_sukro = mean_time_sukro + prod_time/mc_it;
    
    % ---- MODE TIME ----
    for mode = 1:length(n)
    X_unfolded = unfold(X,mode);
    tic
        Y = D_terms{mode,1}*X_unfolded; % multiplication
%         Y = D_terms{mode,1}*unfold(X,mode); % multiplication
    prod_time = toc;
    mean_time_mode(mode) = mean_time_mode(mode) + prod_time/mc_it;
    end
    
    
end
RC_empirical = mean_time_sukro/mean_time_dense
RC_theoretical

for mode = 1:length(n)
    RC_empirical_mode(mode) = mean_time_mode(mode)/mean_time_dense;
    RC_theoretical_mode(mode) = (n(mode)*prod(m))/(prod(n)*prod(m));
end
RC_empirical_mode
RC_theoretical_mode

% Test profiling
    Y = zeros(n);
    profile on;
    for r = 1:cpd_rank
%         Y = Y + tmprod(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + modeprod(X,D_terms(:,r),1:length(m),m); % multiplication
        Y = Y + modeprod3(X,D_terms{1,r},D_terms{2,r},D_terms{3,r},m); % multiplication
%         Y = Y + ttm(X,D_terms(:,r),1:length(m)); % multiplication
    end
    profile off
    profsave(profile('info'),'myprofile_results')

% product D^T*x