% Generate Coefficient Matrix 
% Sirisha Rambhatla, April 2018
function[X] = simple_gen_coeff_mat_X_upto(m, k, p, C)

% k - sparse with |Xi| in [1 C]

% Generate list of indices which will be non zero according to sparsity
nnz_X_orig = zeros(k,p);

% Generate coefficient matrix X
X = zeros(m, p);

% Pick k distinct indices for each sample p, these will be the locations
% of the non zeros
for i = 1:p  
    
    % Select the number of non-zeros and select a subset of non-zero
    % entries
    
    k_max = randi([1,k]);
    shuff = randperm(m, k_max);

    
    % Generate a matrix (k times p) as per distribution D (for non-zero elements) 
    r = C*(-1 + 2*(randn(k_max,1)>0));
    
    % Put these values in the locations picked earlier
    X(shuff,i ) = r;
   
end


end