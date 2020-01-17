%     Copyright 2009, Laurent Jacob, Guillaume Obozinski, Jean-Philippe Vert
%
%     This file is part of overlasso.
% 
%     Overlasso is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     Overlasso is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with overlasso.  If not, see <http://www.gnu.org/licenses/>.


function BW = blocktseng(f, fH, BW0, data, lambda, d, options)

% Laurent Jacob, Feb. 2009
% Inspired by G. Obozinski, Apr. 2008
%
% This code comes with no guarantee or warranty of any kind.
%

% This function minimizes Loss(Y,XW+B) + lambda*||W||_{1,2} using the
% algorithm of Tseng et al. 
% 
% Input: BW0 = initial iterate
%        f = loss function,
%            the calling sequence for f should be
%            [fout,gout]=f(x) where fout=f(x) is a scalar
%              and gout = grad f(x) is a COLUMN vector
%        fH = loss hessian approximation (see Tseng et al.)
%        lambda is the regularization coefficient
%        d is a vector of weights to change the penalty of each group.
% Output: BW = [B; W], where W is the vector of parameters p
% B is the set of non-regularized bias

B = BW0(1);
W = BW0(2:end);
groups = data.groups;
maxit = options.maxit;
display = options.display;
tol = options.tol;
tol_dJ = sqrt(tol)/100;
Jold = Inf;

S=length(groups);
ngc=Inf*ones(S+1,1);     % norms of the gradients of active features
Wnorms = zeros(S,1);
dJ = Inf*ones(S+1,1);

for s=1:S
    Wnorms(s) = norm(W(groups{s}));
end

itc=0;  % Iteration counter

%% MAIN LOOP
while(any(ngc > tol) && itc < maxit && sum(dJ) > tol_dJ)
    s = mod(itc,S+1);
    if ~s % intercept case
        [fc0,gc0]=feval(f, B, W, s, data, 0);
        ngc(s+1) = norm(gc0);   % Recalculating the gradient now that the step has been taken
        hess=feval(fH,B,W,0,data);
        xc = B;
        B = tseng_newton(xc,f,B,W,1,hess,gc0,fc0,0,lambda,data);  % Tseng linesearch
    else  % General case of a feature
        [fcs,gcs]=feval(f, B, W, s, data, 1);
        if norm(gcs) <= d(s)*lambda   % Testing if that block of coordinates should be set to 0
           W(groups{s}) = 0;
            ngc(s+1)=0;
        else
            [fcs,gcs]=feval(f, B, W, s, data, 0);% If not, calculating the gradient
            nws = norm(W(groups{s}));
            if nws ~= 0
                w_nw=W(groups{s})/nws;
                ngc(s+1)=norm(lambda*d(s)*w_nw+gcs);
            else
                ngcs_curr=norm(gcs);
                if ngcs_curr > 0
                    ngc(s+1) = norm(gcs*(1-lambda*d(s)/ngcs_curr));
                else
                    ngc(s+1) = 0;
                end
            end
            hess = feval(fH,B,W,s,data);  % Evaluation of the hessian at the current block
            xc = W(groups{s});  % current block
            W(groups{s}) = tseng_newton(xc,f,B,W,d(s),hess,gcs,fcs,s,lambda,data);  %% Tseng linesearch
         end
         Wnorms(s) = norm(W(groups{s}));
    end
    itc=itc+1;
    if ~S
        J = feval(f, B, W, 0, data, 0);
    else
        J = feval(f, B, W, 0, data, 0) + lambda*d*Wnorms;
    end
    dJ(s+1) = Jold - J;
    Jold = J;
    %if ~mod(itc-1,display)
    %    fprintf('[blocktseng] Iteration %d, J=%g, l2crit=%g, dJ=%g\n',itc,J,ngc'*ngc,sum(dJ));
    %end
end
if itc == maxit
    status = 'max_iter';
elseif all(ngc < tol)
    status = 'gradient_tol';
else
    status = 'fun_tol';
end
%fprintf('[blocktseng] Optimization ended with status %s, J=%g, itc=%d, l2crit=%g, dJ=%g\n',status,J,itc,ngc'*ngc,sum(dJ));
BW = [B; W];
