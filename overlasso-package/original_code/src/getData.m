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

function [data, splits] = getData(type,nfolds,dparam)

switch type
    case 'CHAIN'
        rand('twister',sum(100*clock));
        ntr = dparam.n(1);
        nts = dparam.n(2);
        n = nfolds*(ntr+nts);
        % Build w
        supp = dparam.supp;
        d = dparam.d;
        w = zeros(d,1);
        X = randn(n,d);
        w(supp) = randn(length(supp),1);
        b = randn(1,1);
        bw = [b; w];
        % Center/reduce data
        X = X - repmat(mean(X),n,1);
        X = X*(diag(1./std(X)));
        Y = [ones(n,1) X]*bw;
        % Noise data
        sn2 = dparam.sn2 * mean(Y)^2;
        Y = Y + sn2*randn(n,1);
        data.X = X;
        data.Y = Y;
        splits = cell(1,nfolds);
        for ifold=1:nfolds
            fstart = 1+(ifold-1)*(ntr+nts);
            splits{ifold}.train = fstart:(fstart+ntr-1);
            splits{ifold}.test = (fstart+ntr):(fstart+ntr+nts-1);
        end
        data.bw = bw;
        if strcmp(dparam.grtype,'all')
             data.groups = chaintogroupsall(d,dparam.k);
        else
             data.groups = chaintogroups(d,dparam.k);
        end
        data.dparam = dparam;
        data.entrez=[];
    case 'VV'
        load('../data/vant','X','Y','entrez')

        Y = Y(:,2);
        
        supersplits = cvsplitBal(Y,nfolds(1),fix(100*sum(clock)));
        for i=1:nfolds(1)            
            subsplits{i} = cvsplitBal(Y(supersplits{i}.train),nfolds(2),fix(100*sum(clock)));
        end        
        data.X = X;
        data.Y = Y;
        data.entrez = entrez;
        splits.splits = supersplits;
        splits.subsplits = subsplits;
    otherwise
        error('Unknown dataset')
end


