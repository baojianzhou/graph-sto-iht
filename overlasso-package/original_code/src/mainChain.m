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

function [ocvRes bcvRes data splits] = mainChain(ntr)

% Runs the experiment on synthetic data where the prior is a chain on the
% variables.

addpath(genpath('overlib')) % Overlap group lasso lib
addpath(genpath('blasso'))  % Lasso lib (faster)

%% Parameters

% Data
DATASET = 'CHAIN';
preproc = 'plain'; % plain|correl

LOSS = 'l2';

% Number of folds
nfolds = 1;

% Toy parameters
dparam.d = 100;
dparam.grtype = 'none';
dparam.k = 2;
dparam.supp = [5:24 90:92];%(20:40);
dparam.sn2 = 0.1; % Noise on data
dparam.n = [str2double(ntr) 500]; % Number of train/test points


%% Experiments

[data, splits] = getData(DATASET,nfolds,dparam);

klist = [2 4 6]; % Sizes of the linear subgraphs to be tested.

ocvRes = cell(1,length(klist));

tic
for i=klist
    data.groups = chaintogroups(dparam.d,i); % The graph prior is encoded by the groups (chains).
    ocvRes{find(i==klist)} = runMethod(data, splits, 'over', preproc, LOSS);
end
bcvRes = runMethod(data, splits, 'blasso', preproc, LOSS);
toc

