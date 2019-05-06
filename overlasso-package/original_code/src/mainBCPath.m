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

function [res data pathways splitsstr] = mainBCPath()

% Breast Cancer data with pathway-based penalty

addpath(genpath('blasso'))
addpath(genpath('overlib'))

%% Parameters

% Data
DATASET = 'VV';
% Loss function
LOSS = 'blr'; % l2 | lr | blr
% Which preprocessing apply to X
preproc = 'correl'; % plain|correl
dparam = 0;

% Number of folds
nfolds = [5 5];


%% Experiments

[data, splitsstr] = getData(DATASET,nfolds,dparam);

fprintf('[mainBCPath] Data formated\n');

pathways=load('../data/groupings/pathways.txt');
% restrict to used entrez
pathways = pathways(ismember(pathways(:,2),data.entrez),:);

% Keep only genes that are in at least one pathway
varInPath = find(ismember(data.entrez,unique(pathways(:,2))));
data.X = data.X(:,varInPath);
data.entrez = data.entrez(varInPath);

grids = unique(pathways(:,1))';

pgroups = cell(1,length(grids));
c = 0;
for g=grids
    c = c + 1;
    centrez = pathways((pathways(:,1) == g),2);
    pgroups{c} = find(ismember(data.entrez,centrez))';
end

res = cell(1,2);

% Only use the first fold
%splitsstr.splits = {splitsstr.splits{1}};
%splitsstr.subsplits = {splitsstr.subsplits{1}};

lambdas = fliplr(2.^(-12:0.1:0));
%lambdas = fliplr(2.^(-8:2:0));

tic
% Lasso penalty
fprintf('[mainBCGr] Starting lasso\n');
data.groups = num2cell(1:size(data.X,2));
res{1} = cvOverLasso(data,splitsstr,lambdas,preproc, LOSS);
% Groups are the pathways
fprintf('[mainBCGr] Starting structure lasso on pathways\n');
data.groups = pgroups;
res{2} = cvOverLasso(data,splitsstr,lambdas,preproc, LOSS);
toc

%save(sprintf('../res/mainBCPath-%s',dataset),'res','pathways');
