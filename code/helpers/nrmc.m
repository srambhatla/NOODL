function [A_n] = nrmc(A)
% Alternative to normc for Matlab 2014b and above. 

A_n = zeros(size(A));

    for i = 1:size(A, 2)
       if (sum(A(:,i)) ~= 0)
         A_n(:,i) = A(:,i)/norm(A(:,i));
       end
    end
end