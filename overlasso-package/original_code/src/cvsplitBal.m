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

function splits = cvsplitBal(y,nfolds,seed)

% Same as cvsplit, but splits the positive and the negative data separately
splits = cell(1,nfolds);

posidx = find(y == 1)';
negidx = find(y == -1)';

np = length(posidx);
nn = length(negidx);

psizefold = round(np/nfolds);
nsizefold = round(nn/nfolds);

if nargin>2
    rand('seed',seed);
end

pa = posidx(randperm(length(posidx)));
na = negidx(randperm(length(negidx)));

for ifold = 1:nfolds
    % prepare the fold
    if ifold < nfolds
        splits{ifold}.train = [pa([1:(ifold-1) * psizefold , 1+(ifold * psizefold):np]) na([1:(ifold-1) * nsizefold , 1+(ifold * nsizefold):nn])];
        splits{ifold}.test = [pa((1+(ifold-1) * psizefold):(ifold * psizefold)) na((1+(ifold-1) * nsizefold):(ifold * nsizefold))];
    else
        splits{ifold}.train = [pa(1:(ifold-1) * psizefold) na(1:(ifold-1) * nsizefold)];
        splits{ifold}.test = [pa((1+(ifold-1) * psizefold):np) na((1+(ifold-1) * nsizefold):nn)];
    end
end


