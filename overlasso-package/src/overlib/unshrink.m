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


function [BW res] = unshrink(data, tridx, tsidx, AS, BW0)

f=@lrOracle;
fH=@lrH;

options.display = 100;
options.maxit=2e4;
options.tol = 1e-4;

lambda = 1e-8;

tsdata.X = [ones(length(tsidx),1) data.X(tsidx,:)];
tsdata.Y = data.Y(tsidx);

sidx = [];
for g=AS
    sidx = [sidx data.groups{g}];
end

trdata.X = data.X(tridx,sidx);
trdata.Y = 1 + (1+data.Y(tridx))/2;
trdata.groups = cell(1,length(AS));

fprintf('[unshrink] Re-running optimization with %d groups in the active set and little constraint\n',length(AS));
%[BW, res.dev, res.stats] = mnrfit(trdata.X,trdata.Y,'estdisp','on');
BW = blocktseng(f, fH, BW0, trdata, lambda, ones(1,length(AS)), options);
[res.perf res.pred] = test(tsdata,BW,'class');
res.trauc = auc(BW(1)+(trdata.X*BW(2:end)),trdata.Y);
res.tsauc = auc(res.pred,tsdata.Y);
