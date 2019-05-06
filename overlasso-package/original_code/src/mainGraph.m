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

function [res data] = mainGraph(ntr)

% Runs the experiment on synthetic data where the prior is a grid on the
% variables.

addpath(genpath('blasso'))
addpath(genpath('overlib'))

%% Parameters

% Data
DATASET = 'CHAIN';
% Which preprocessing apply to X
preproc = 'plain'; % plain|correl

LOSS = 'l2';

% Number of folds
nfolds = 1;

% Toy parameters
dparam.width = 10;
dparam.height = 10;
dparam.d = dparam.width*dparam.width;
dparam.supp = [34:37 44:47 54:57 64:67];
dparam.grtype = 'none';
dparam.k = 2;
dparam.sn2 = 0.1; % Noise on data
dparam.n = [str2double(ntr) 100]; % Number of train/test points

%% Experiments

[data, splits] = getData(DATASET,nfolds,dparam);

%edges = buildGrid(dparam.width,dparam.height);

 % The graph prior is encoded by the groups (squares).
data.groups = buildGridSquares(dparam.width,dparam.height);

res = cell(1,2);

res{1} = runMethod(data, splits, 'blasso', preproc, LOSS);
%data.groups = edgestogroups(edges,2);
res{2} = runMethod(data, splits, 'over', preproc, LOSS);

