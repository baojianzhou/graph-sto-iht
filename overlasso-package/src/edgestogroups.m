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

function groups = edgestogroups(edges , k)

% function groups = edgestogroups(k)
%
% edges is a p*2 matrix of pairwise relationships (edges)
% k is the size of the groups we want

nedges = size(edges,1);
% Put all singletons
nvertices = max(edges(:));
for i=1:nvertices
    groups{i} = i;
end

% also add edges
if k>1
    for i=1:nedges
        groups{nvertices+i} = edges(i,:);
    end
end