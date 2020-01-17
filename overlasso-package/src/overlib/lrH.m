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


function H =lrH(b,w,s,data)

% Computes an approximation of the hessian of the (unbalanced) logistic
% regression function.

% INPUT:
%     - b is the offset, w the linear function.
%     - s is the index of a group (0 for the offset).
%     - data contains X, Y and groups.
% OUTPUT:
%     - H is the hessian approximation.

x = data.X;
y = data.Y;
groups = data.groups;

if isempty(w)
    expterm = exp(-y * b);
else
    expterm = exp(-y .* (x*w + b));
end
n = length(y);

if s
    % Diag of block hessian
    diagH = (x(:,groups{s}).^2)'*(expterm ./((1+expterm).^2))/n;

    % Block hessian approximation
    H = diag(diagH);
else
    H = sum(expterm ./((1+expterm).^2))/n;
end