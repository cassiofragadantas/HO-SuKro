%% Visualizing
% Show original terms - absolute values are used to avoid signal
% ambiguity, since there is a multiplicative coefficient ambiguity.
for k = 1:length(n)
    for r = 1:alpha
        subplot(length(n),alpha,r + (k-1)*alpha);
        imshow(abs(D_terms{k,r,:}),[]);
    end
end
suptitle('Original terms')

% Find the best correspondance between the recovered terms wrt the
% original terms.
term = Uhat{1};
var_vec = zeros(1,alpha);
order = zeros(1,size(Uhat{1},2));
% for r = 1:size(Uhat{1},2)
%     for r_orig = 1:alpha
%         element_div = D_terms{end,r_orig,:}./reshape(term(:,r),n(end),m(end));
%         var_vec(r_orig) = var(element_div(:));
%     end
%     [~, order(r)] = min(var_vec);
% end
% Disable correspondance search
order = 1:size(Uhat{1},2);

% Show recovered terms
figure;
for k = 1:length(n)
    for r = 1:size(Uhat{1},2)
        subplot(length(n),alpha,order(r) + (k-1)*size(Uhat{1},2)); % ERROR if size(Uhat{1},2) > alpha
        term = Uhat{end-k+1};
        imshow(abs(reshape(term(:,r),n(k),m(k))),[]);
    end
end
suptitle('Recovered terms')

% Reconstruct a given rank
% r = 3;
% result = Uhat{end};
% result = reshape(result(:,r),n(1),m(1));
% for k = 2:length(n)
%     term = Uhat{end-k+1};
%     term = reshape(term(:,r),n(k),m(k));
%     result = kron(result,term);
% end