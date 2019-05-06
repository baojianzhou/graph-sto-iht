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


function [f,gradf]=lrOracle(b,w,s,data,atZero)

% Compute the (unbalanced) logistic regression function and its gradient

% INPUT:
%     - b is the offset, w the linear function.
%     - s is the index of a group (0 for the offset)
%     - data contains X, Y and groups.
%     - atZero: if 1, the loss will be computed for the same w, but with
%     the group s set to 0.
% OUTPUT:
%     - f is the value of the function.
%     - grad is a column vector containing the partial gradient with
%     respect to group s.

x = data.X;
y = data.Y;
groups = data.groups;
if isempty(w)
    expterm = exp(-y * b);
else
    wloc = w;
    if atZero % Evaluate loss/grad for w(group)=0
        wloc(groups{s}) = 0; 
    end
    expterm = exp(-y .* (x*wloc + b));
end

n = length(y);

% Loss
f = sum(log(1+expterm))/n;

% Block gradient
if s
    gradf = (x(:,groups{s})'*(-y .* expterm ./ (1+expterm)))/n;
else
    gradf = sum(-y .* expterm ./ (1+expterm))/n;
end