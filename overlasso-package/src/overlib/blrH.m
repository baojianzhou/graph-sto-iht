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


function H = blrH(b,w,s,data,Cp,Cn)

x = data.X;
y = data.Y;
groups = data.groups;

% % Balanced loss
n = length(y);
posidx = find(y == 1);
negidx = find(y == -1);
xp = x(posidx,:);
xn = x(negidx,:);
yp = y(posidx);
yn = y(negidx);
% Loss
if isempty(w)    
    pexpterm = exp(-yp * b);
    nexpterm = exp(-yn * b);
else
    pexpterm = exp(-yp .* (xp*w + b));
    nexpterm = exp(-yn .* (xn*w + b));
end

if s
    % Diag of block hessian
    diagH = (Cp*(xp(:,groups{s}).^2)'*(pexpterm ./((1+pexpterm).^2)) + Cn*(xn(:,groups{s}).^2)'*(nexpterm ./((1+nexpterm).^2)))/n;

    % Block hessian approximation
    H = diag(diagH);
else
    H = (Cp*sum(pexpterm ./((1+pexpterm).^2)) + Cn*sum(nexpterm ./((1+nexpterm).^2)))/n;
end
