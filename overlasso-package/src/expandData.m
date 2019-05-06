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

function edata = expandData(data)

x = data.X;
groups = data.groups;
p = size(x,2);

% First get the size of the new matrix
ep = 0;
for g=1:length(groups)
    ep = ep+length(groups{g});
end

%ex = [];
%ex = zeros(size(x,1),es);
egroups = cell(1,length(groups));
%exp2orig = [];
exp2orig = zeros(p,ep);
c=0;
start=1;
for g=1:length(groups)    
    egroups{g} = start:(start+length(groups{g})-1);
    %ex = [ex x(:,groups{g})];
    %ex(:,egroups{g}) = x(:,groups{g});
    start = start+length(groups{g});
    for i=groups{g}
        c=c+1;
        exp2orig(:,c) = ismember(1:p,i)';
    end
end

ex = x*exp2orig;

edata = data;
edata.X = ex;
edata.groups = egroups;
edata.exp2orig = exp2orig;
