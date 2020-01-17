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

function splits = cvsplit(n,nfolds,seed)

%
% function splits = cvsplit(n,nfolds,seed)
%
% Prepare training and test splits of nfolds-fold cross-validation
%
% INPUT
% - n is the total number of samples
% - nfolds is the number of folds
% - seed is the random number generator seed (optional)
%
% OUTPUT
% - splits is an array with components s{i}.train and s{i}.test, for 1 <= i
% <= nfolds, that lists the indexes in the training and test sets for fold
% i.

    sizefold = round(n/nfolds);
    if nargin>2
        rand('seed',seed);
    a = randperm(n);
    for ifold = [1:nfolds]
        % prepare the fold
        if ifold < nfolds
            splits{ifold}.train = a([1:(ifold-1) * sizefold , 1+(ifold * sizefold):n]);
            splits{ifold}.test = a([(1+(ifold-1) * sizefold):(ifold * sizefold)]);
        else
            splits{ifold}.train = a([1:(ifold-1) * sizefold]);
            splits{ifold}.test = a([(1+(ifold-1) * sizefold):n]);
        end
    end
end

