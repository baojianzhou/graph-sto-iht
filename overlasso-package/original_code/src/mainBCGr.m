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

function [res data] = mainBCGr()

% Breast Cancer data with Graph-based penalty

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

fprintf('[mainBCGr] Data formated\n');

edges = load('../data/ideker/edge.txt');

varInGraph = unique(edges(:));
data.X = data.X(:,varInGraph);
data.entrez = data.entrez(varInGraph);

res = cell(1,2);

% Only use the first fold
%splitsstr.splits = {splitsstr.splits{1}};
%splitsstr.subsplits = {splitsstr.subsplits{1}};

%lambdas = fliplr(2.^(-12:0.1:0));
lambdas = fliplr(2.^(-8:2:0));

tic
% Lasso penalty
fprintf('[mainBCGr] Starting lasso\n');
data.groups = num2cell(1:size(data.X,2));
res{1} = cvOverLasso(data,splitsstr,lambdas,preproc,LOSS);
% Groups are the edges
fprintf('[mainBCGr] Starting structure lasso on edges\n');
data.groups = edgestogroups(edges,2);
res{2} = cvOverLasso(data,splitsstr,lambdas,preproc,LOSS);
toc

% save(sprintf('../res/mainBCGr-%s',dataset),'res');
