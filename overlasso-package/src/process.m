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

function [pdata kidx,kgroups,kgroupidx] = process(data,type)

switch type
    case 'correl'
        p = size(data.X,2);
        r = zeros(1,p);
        for i = 1:p
            C = corrcoef(data.X(:,i),data.Y);
            r(i) = C(1,2);
        end
        [ts,idx]=sort(abs(r),'descend');
        kidx = idx(1:500);
%        kidx = idx(1:round(0.5*size(data.X,2)));
%        kidx = find(abs(r) > 0.25);
        pdata.Y = data.Y;
        pdata.X = data.X(:,kidx);
        pdata.entrez = data.entrez(kidx);
    case 'threshold'
        kidx = find(max(abs(data.X) > 0.2));
        pdata.Y = data.Y;
        pdata.X = data.X(:,kidx);
        pdata.entrez = data.entrez(kidx);
    otherwise
        error('Unknown preprocessing')
end

% Restrict groups to remaining variables, and convert variable numbers.
groups = data.groups;
c = 0;
for g =1:length(groups)
    rv = groups{g}(ismember(groups{g},kidx));
    if ~isempty(rv)
        c = c+1;
        tmp = zeros(1,length(rv));
        i = 0;
        for v=rv
            i = i+1;
            tmp(i) = find(kidx == v);
        end
        pgroups{c} = tmp;
        kgroups{c} = rv;
        kgroupidx(c) = g;
    end     
end

pdata.groups = pgroups;
fprintf('[process] Pre-process over. Kept %d(/%d) variables and %d(/%d) groups\n',length(kidx),size(data.X,2),length(pgroups),length(groups));

