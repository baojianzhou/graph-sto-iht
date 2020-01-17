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

function fcycles = buildGridSquares(h,w)

% function fcycles = buildGridSquares(h,w)
% Builds 4-cycles for a grid of height h, width w.

n = h*w;

fcycles = {};

for i=1:n
    % Upper right cycle
    if mod(i,w) && i>w
        c = [i i+1 i-w i-w+1];
        fcycles = addCycle(c,fcycles);
    end
    % Upper left cycle
    if mod(i-1,w) && i>w
        c = [i i-1 i-w i-w-1];
        fcycles = addCycle(c,fcycles);
    end
    % Bottom left cycle
    if mod(i-1,w) && i <= w*(h-1)
        c = [i i-1 i+w i+w-1];
        fcycles = addCycle(c,fcycles);
    end
    % Bottom right cycle
    if mod(i,w) && i <= w*(h-1)
        c = [i i+1 i+w i+w+1];
        fcycles = addCycle(c,fcycles);
    end
end

function fcycles = addCycle(c,fcycles)
found = 0;
for k=1:length(fcycles)
    if all(ismember(c,fcycles{k}))
        found = 1;
        break;
    end
end
if ~found
    fcycles{length(fcycles)+1} = c;
end
