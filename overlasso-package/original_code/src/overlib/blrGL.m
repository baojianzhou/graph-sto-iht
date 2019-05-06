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


function gradf = blrGL(b,w,data,Cp,Cn)

x = data.X;
y = data.Y;


% Balanced loss
n = length(y);
posidx = logical(y == 1);
negidx = logical(y == -1);

% Loss
if isempty(w)
    expterm = exp(-y * b);
else
    expterm = exp(-y .* (x*w + b));
end
y(posidx) = Cp*y(posidx);
y(negidx) = Cn*y(negidx);
gradf = x'*(-y .* expterm ./ (1+expterm))/n;
