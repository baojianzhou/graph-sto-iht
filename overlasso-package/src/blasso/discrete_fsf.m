function result = discrete_fsf(y, dictionary, eps_step)
% =========================================================================
% function result = discrete_fsf(y, x, eps_step)
% =========================================================================
% PURPOSE: 
% This function calculates the regularization path for the discrete steps
% FSF algorithm (Efron et. al. (2004)). 
% =========================================================================
% INPUTS:
%  y is the dependent variable
%  x is a matrix with explanatory variables in its columns
%  eps_step is the step size for the algorithm
% =========================================================================
% OUTPUTS:
% A structure result containing the fields:
% 
% nbetas        a ??? x n_variables matrix containing the coefficients for 
%               the normalized regression (both y and x are normalized)
% betas         a ??? x n_variables matrix containing the coefficients for 
%               the unnormalized regression
% intercepts    a ??? dimensional vector containing the intercepts for the 
%               unnormalized regression
% yscale        the factor used in the normalization of y
% ymean:        the mean of the dependent variable
% xscale:       the factor used in the normalization of x       
% xmean:        the means of the columns of x
% normalized_y: a normalized version of the input argument y
% original_y:   the input argument y
% normalized_x: a normalized version of the input argument x
% original_x    the input argument x
% =========================================================================
% KNOWN ISSUES:
% None at this time
% =========================================================================
% Send comments, suggestions, bug reports to Guilherme Rocha: 
% gvrocha [AT] gmail [dot] com
% =========================================================================
% See also LASSO, BLASSOL2


% 0. Setting parameters and starting up some variables
% =========================================================================
zero_eps     = 1e-6;                                                       % numbers smaller than this will be considered zero
n_obs        = size(dictionary, 1);                                        % this is the number of observations

% 0.1 Normalizing variables: 
% =========================================================================
warning off;
[ndictionary, xmeans, xscale]  = normalize_matrix(dictionary);             % normalizing the regressors: warning off as some of them may have zero scale
warning on;
[ny, ymeans, yscale]           = normalize_matrix(y);                      % normalizing the dependent variable: not necessary but may be helpful in "standardizing" the step size

constants_idx = find(xscale < zero_eps);                                   % variables that are actually "constants"
variables_idx = find(xscale >= zero_eps);                                  % actual variables (they DO vary)

zero_signal = ndictionary(:,constants_idx);
ndictionary = ndictionary(:,variables_idx);
unit_vectors = eye(size(ndictionary,2));                                   % this is a matrix with unit vectors in its columns - need to change this

xtx         = ndictionary'*ndictionary;                                    % xtx is the correlation matrix of the explanatory variables
xtr         = ndictionary'*ny;                                             % xty is a vector with the correlations between y and the explanatory variables

% 1. Initializing path:
% =========================================================================
curr_lambda   = Inf;                                                         % Path starts with lambda = +infinity ...
curr_beta     = zeros(size(ndictionary,2),1);                                % and beta = 0
betas         = curr_beta;                                                   % betas is a matrix containing the selected betas as its columns
lambdas       = curr_lambda;                                                 % lambdas is a vector contatining the values of the multipliers for which the columns in betas are an approximation to the optimal value
n_lambdas     = 1;                                                           % n_lamdas is the length of the lambdas vector and the number of columns in betas
backward_flag = [];                                                          % keep track of backward steps;

% 3. While improvement is possible
% =========================================================================
loss_improv = Inf;
while(loss_improv > zero_eps)  

  % 3.1 Find best direction to move
  % =====================================================================
  [cj, j]           = max(abs(xtr));                                     % Find direction to move
  sj                = sign(xtr(j));                                      % Get sign of direction to move
  new_beta          = curr_beta;    
  new_beta(j)       = curr_beta(j) + sj;                                 % Computes the new beta (in step count units)
  loss_improv       = 2*sj*xtr(j)-eps_step*xtx(j,j);                     % This is the improvement in the loss function
  
  if(loss_improv>zero_eps)
    % 3b.2.1 If lambda decreased and improvement happened: relax constraint
    % =====================================================================
    betas           = [betas new_beta];                                  % - add the new estimate to the path
    n_lambdas       = n_lambdas + 1;                                     % - increase the recorded size of the lambda vector
  end;
    
  xtr               = xtr - eps_step*sj*xtx(:,j);                        % Updates correlation between explanatory variables and residuals
  curr_beta         = new_beta;                                          % Updates the vector of estimates
end;

% 4. Saving the outputs (path included) in a structure:
% =========================================================================
result.nbetas(:,variables_idx) = eps_step*betas';
result.nbetas(:,constants_idx) = zeros(size(betas, 2), size(constants_idx, 2));
result.betas(:,variables_idx)  = yscale*eps_step*betas'*diag(1./xscale(variables_idx));
result.betas(:,constants_idx)  = zeros(size(betas, 2), size(constants_idx, 2));
result.intercepts    = ymeans-(xmeans*result.betas')';
result.yscale        = yscale;
result.ymean         = ymeans;
result.xscale        = xscale;
result.xmean         = xmeans;
%
result.normalized_y  = ny;
result.original_y    = y;
result.normalized_x(:,constants_idx)  = zero_signal;
result.normalized_x(:,variables_idx) = ndictionary;
result.original_x    = dictionary;
result.fitted        = kron((result.intercepts'), ones(n_obs,1)) + dictionary*result.betas';
result.residuals     = kron(y, ones(1, size(result.betas, 1))) - result.fitted;
result.sample_size   = n_obs;
result.L1_nbetas     = sum(abs(result.nbetas'))';
result.L1_betas      = sum(abs(result.betas'))';
result.method        = 'fsf';

% -------------------------------------------------------------------------
% USES: 
% - normalize_matrix
% -------------------------------------------------------------------------
% LOG: 
% 2007/05/11: function created: from blassol2
% -------------------------------------------------------------------------


function [x, means, scales] = normalize_matrix(x)
%--------------------------------------------------------------------------
% function [x, means, scales] = normalize_matrix(x)
%--------------------------------------------------------------------------
% PURPOSE:
% This function normalizes the columns of matrix x to have mean zero and
% sum of squares equals to 1
%--------------------------------------------------------------------------
% INPUTS:
% x: a nxp matrix whose columns are to be normalized
%--------------------------------------------------------------------------
% OUTPUTS:
% x:      matrix x normalized to have \sum_{j=1}^{n}x_{ji}^{2} = 1 and 
%         mean of x(:,i) = 0, \forall i between 1 and p
% means:  a 1 x p vector with the means of matrix x
% scales: a 1 x p vector with the sum of squares of the demeaned by columns
%         x matrix
%--------------------------------------------------------------------------
% KNOWN BUGS/LIMITATIONS:
% - This function will not deal well with missing values
%--------------------------------------------------------------------------
ncols  = size(x,2);
nobs   = size(x,1);
scales = zeros(1,ncols);
means  = mean(x);
for i = 1:ncols
  x(:,i)    = x(:,i)-means(i);
  scales(i) = sqrt(sum(x(:,i).^2));
  x(:,i)    = x(:,i)/scales(i);
end;

% -------------------------------------------------------------------------
% USES: 
% - size
% - zeros
% - mean
% - sqrt
% -------------------------------------------------------------------------
% LOG: 
% 2005/08/24: function commented
% -------------------------------------------------------------------------