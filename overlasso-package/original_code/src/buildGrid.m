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

function edges = buildGrid(h,w)

% function A = buildGrid(h,w)
% Builds edges for a grid of height h, width w.

n = h*w;

edges = [];

for i=1:n
    % Link right node 
    if mod(i,w)
        e = [i i+1];
        edges = addEdge(e,edges);
    end
    % Link left node
    if mod(i-1,w)
        e = [i i-1];
        edges = addEdge(e,edges);
    end
    % Link node below
    if i <= w*(h-1) 
        e = [i i+w];
        edges = addEdge(e,edges);
    end
    % Link node above
    if i > w
        e = [i i-w];
        edges = addEdge(e,edges);
    end
end


% Building the sif file
fid = fopen('grid.sif','wt');     % 'wt' means "write text"
if (fid < 0)
    error('could not open file "network.sif"');
end;
for i=1:n
    fprintf(fid, '%d \n', i);
end
for i=1:size(edges,1)
    fprintf(fid, '%d pp %d \n', edges(i,1), edges(i,2));
end
fclose(fid);


function edges = addEdge(e,edges)
if isempty(edges) || all(any(edges ~= repmat(fliplr(e),size(edges,1),1),2));
    edges = [edges; e];
end
