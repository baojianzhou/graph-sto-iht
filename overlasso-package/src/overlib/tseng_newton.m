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


function [xt]=tseng_newton(xc,f,b,w,ds,Jh,Jg,Jfc,s,lambda,data)

% This function just implements the optimization of Tseng et al. on a given
% group.

thresh=1e-1;
coeff=0.5;
maxitr=30;
groups = data.groups;
weval = w;

if s % Case of a feature (not the offset)
    % Compute the direction
    if norm(xc)==0 && norm(Jg)<ds*lambda
        xt=xc;
        return
    end
    hmax=max(max(diag(Jh)),1e-16);
    u = Jg-hmax*xc;
    if norm(u)<ds*lambda
        d = -xc;
    else
        u = u/norm(u);
        d = -(Jg-lambda*ds*u)/hmax;
    end
    alpha=1000;
    Jgd=Jg'*d;
    if norm(xc)>0,
        gcd=Jgd+lambda*ds*(xc'/norm(xc))*d;
    else
        gcd=Jgd;
    end
    if gcd>0
        error('d is not a descent direction');
    end
    % Linesearch with Armijo condition
    nxc=norm(xc);
    delta=Jgd+lambda*ds*(norm(xc+d)-nxc);
    itr=1;
    xt=xc+alpha*d;
    weval(groups{s}) = xt;
    Jft=feval(f,b,weval,s,data,0);
    wolfe=logical(Jft-Jfc+lambda*ds*(norm(xt)-nxc)<=alpha*thresh*delta);
    while(~wolfe && itr < maxitr)
        alpha=alpha*coeff;
        xt=xc+alpha*d;
        weval(groups{s}) = xt;
        Jft=feval(f,b,weval,s,data,0);
        wolfe=logical(Jft-Jfc+lambda*ds*(norm(xt)-nxc)<=alpha*thresh*delta);
        itr=itr+1;
    end
else % Case of the offset
    % Compute the direction
    d=-Jh\Jg;
    alpha=1;
    Jgd=Jg'*d;
    if Jgd>0
        error('d is not a descent direction');
    end
    % Linesearch with Armijo condition
    delta=Jgd;
    itr=1;    
    xt=xc+alpha*d;
    Jft=feval(f,xt,w,s,data,0);
    wolfe=logical(Jft-Jfc <= (alpha*thresh*delta));
    while(~wolfe && (itr < maxitr)),
        alpha=alpha*coeff;
        xt=xc+alpha*d;
        Jft=feval(f,xt,w,s,data,0);
        wolfe=logical(Jft-Jfc <= (alpha*thresh*delta));
        itr=itr+1;
    end
end

if itr==maxitr
    fprintf('Maximal number of iterations reached in Armijo\n');
end
