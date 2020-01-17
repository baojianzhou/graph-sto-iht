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

function [err,ypred] = test(data,w,loss)

%global data;

x = data.X;
y = data.Y;

switch loss
    
    case 'l2'
        
        ypred = x*w;
        diff = ypred - y;
        err = diff'*diff/length(y);
        
    case 'l2Bal'
        
        posidx = find(y<0);
        negidx = find(y>0);

        cp = length(negidx)/length(y);
        cn = length(posidx)/length(y);

        yp = y(posidx);
        yn = y(negidx);
        
        ypred = x*w;
        
        ypp = ypred(posidx);
        ypn = ypred(negidx);
        
        diffp = ypp - yp;
        diffn = ypn - yn;
        
        err = cp*diffp'*diffp + cn*diffn'*diffn;
        
    case 'class'

        ypred = x*w;
        err = sum(y ~= sign(ypred))/length(y);
        
    case 'classBal'
        
        posidx = find(y>0);
        negidx = find(y<0);

        
        ypred = x*w;
        
%        ep = cp*sum(sign(ypred(posidx)) ~= 1)/length(posidx);
%        en = cn*sum(sign(ypred(negidx)) ~= -1)/length(negidx);
        
        err = (sum(sign(ypred(posidx)) ~= 1)/max(1,length(posidx)) + sum(sign(ypred(negidx)) ~= -1)/max(1,length(negidx)))/2;
        
    case 'all'
        
        posidx = find(y>0);
        negidx = find(y<0);

        ypred = x*w;
        
%        ep = cp*sum(sign(ypred(posidx)) ~= 1)/length(posidx);
%        en = cn*sum(sign(ypred(negidx)) ~= -1)/length(negidx);
        
        err.auc = auc(ypred,y);
        err.acc = sum(y ~= sign(ypred))/length(y);
        err.bacc = (sum(sign(ypred(posidx)) ~= 1)/max(1,length(posidx)) + sum(sign(ypred(negidx)) ~= -1)/max(1,length(negidx)))/2;
        
    otherwise
        
        error('This loss is not implemented')
end
