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


function res = overLasso(data,tridx,tsidx,lambdas, Winit, useAS, gweight, loss)

% function res = overLasso(data,tridx,tsidx,lambdas, Winit, useAS, gweight, loss)
%
% This function optimizes a given loss function with the overlap penalty of
% (Jacob, Obozinski and Vert, 2009). In practice, given the data X, Y and
% the groups of covariates, it creates for each original variable one
% duplicate for each of its occurences in a group, then runs regular group
% lasso on this latent representation.
%
%
% INPUT:
%       - data, a structure with fields 'X', 'Y' and 'groups', the latter
%       being a cell array, such that data.groups{g} contains the indices
%       of the variables belonging to group g.
%       - tridx/tsidx: arrays containing the rows of X and Y to be used as
%       training examples (resp. test).
%       - lambdas: the regularization parameter values for which the
%       optimization should be performed.
%       - Winit: an initialization of [B W], or [].
%       - useAS: 1 to use the active set approach (recommended), 0
%       otherwise.
%       - gweight: 1 to reweight the group norms in the penalty, 0
%       otherwise.
%       - loss: smooth loss function used for learning. Choose among
%       squared loss ('l2'), logistic regression ('lr') and balanced
%       logistic regression ('blr'). Note that the performance measure is automatically 
%       adapted to the type of loss.
%
% OUTPUT:
%       - res, a structure containing the performance for each lambda
%       (perf), the predicted outputs (pred), the used lambdas (lambdas),
%       the active variables at the optimum for each lambda (AS) and the
%       optimal function in both the original (oWs) and expanded (Ws)
%       spaces for each lambda.

unshrink = 0; % Used to run a less penalized learning round on the selected variables as a postprocessing. 0 to desactivate it.

options.display = 1000;
options.maxit=2e4;
options.tol = 1e-5;
options.bias = 0;

% Duplication step
%fprintf('[overLasso] Duplicating variables\n');
edata = expandData(data);
%fprintf('[overLasso] Variables duplicated (from %d to %d)\n',size(data.X,2),size(edata.X,2));

trdata.X = edata.X(tridx,:);
trdata.Y = edata.Y(tridx);
trdata.groups = edata.groups;

tsdata.X = [ones(length(tsidx),1) edata.X(tsidx,:)];
tsdata.Y = edata.Y(tsidx);

[n p] = size(trdata.X);

% Select used function according to the chosen loss
% f/fH/fGL: addresses of functions returning the loss function and
% its (partial) gradient (f), its approximated hessian (fH) and the
% full gradient vector (fGL). In addition, ftest selects the performance
% measure (see test.m).
switch loss
    case 'lr' % Logistic regression
        f = @lrOracle;
        fH = @lrH;
        fGL = @lrGL;
        ftest = 'class'; % Classification error
    case 'blr' % Balanced logistic regression   
        np = sum(trdata.Y == 1);
        nn = sum(trdata.Y == -1);
        n = length(trdata.Y);
        Cp = nn/n;
        Cn = np/n;                     
        f = @(b,w,s,data,atZero) blrOracle(b,w,s,data,atZero,Cp,Cn);
        fH = @(b,w,s,data) blrH(b,w,s,data,Cp,Cn);
        fGL = @(b,w,data) blrGL(b,w,data,Cp,Cn);
        ftest = 'all'; % Return accuracy, balanced accuracy and AUC.
        %ftest = 'classBal';
    case 'l2' % Squared loss
        f = @rrOracle;
        fH = @rrH;
        fGL = @rrGL;
        ftest = 'l2'; % Squared error.
    otherwise
        error('Unknown loss function');
end

% Group weights
d = ones(1,length(trdata.groups));
if gweight
    % Trace version
    for g=1:length(trdata.groups)
        d(g) = sum(svd(trdata.X(:,trdata.groups{g})));
    end
    d = d/max(d);
end

Ws = zeros(p+1,length(lambdas));
ASarr = cell(1,length(lambdas));
completeASarr = cell(1,length(lambdas));

if ~isempty(Winit)
    BW = Winit;
else
    BW = zeros(p+1,1);
end

if ~strcmp(ftest,'all')
    res.perf = -ones(1,length(lambdas));
else
    res.auc = -ones(1,length(lambdas));
    res.acc = -ones(1,length(lambdas));
    res.bacc = -ones(1,length(lambdas));
end
res.pred = zeros(size(tsdata.X,1),length(lambdas));
res.lambdas = lambdas;
res.nextGrad = -ones(1,length(lambdas));

if useAS
    AS = [];
else
    AS = 1:length(trdata.groups);
end
    
options.maxFeat = size(trdata.X,2);

% For each lambda on the grid, calls the optimization function (with warm restart, i.e., starting
% from the previous optimal function/active set).
i=0;
for lambda=lambdas
    i = i+1;
    fprintf('[overLasso] Starting optimization with lambda=%f\n',lambda);
    [BW AS completeAS nextGrad] = blocktsengAS(f, fH, fGL, BW, AS, trdata, lambda, d, options);
    if ~useAS
        AS = 1:length(trdata.groups);
        completeAS = AS;
    end    
    if logical(unshrink)
        % sidx must contain the indices of the variables from all the groups in the
        % active set.
        sidx = [];
        for g=AS
            sidx = [sidx data.groups{g}];
        end
        subdata.X = trdata.X(:,sidx);
        subdata.Y = trdata.Y;
        subdata.groups = cell(1,length(AS));
        c = 0;
        for g=AS
            c = c+1;
            subdata.groups{c} = find(ismember(sidx,trdata.groups{g}));
        end
        sbw = blocktseng(f, fH, [BW(1); BW(1+sidx)], subdata, unshrink, d(AS), options);                
        Ws([1 1+sidx],i) = sbw;
    else
        Ws(:,i) = BW;
    end
    res.nextGrad(i) = nextGrad;
    ASarr{i} = AS;
    completeASarr{i} = completeAS;
    if ~isempty(tsidx)
        if ~strcmp(ftest,'all')
            [res.perf(i) res.pred(:,i)] = test(tsdata,Ws(:,i),ftest);
        else
            [tmp res.pred(:,i)] = test(tsdata,Ws(:,i),ftest);
            res.auc(i) = tmp.auc;
            res.acc(i) = tmp.acc;
            res.bacc(i) = tmp.bacc;
        end
    end
end

if strcmp(ftest,'all')
    res.perf = res.bacc;
end

res.AS = ASarr;
res.completeAS = completeASarr;
res.Ws = Ws;
res.oWs = [Ws(1,:); edata.exp2orig*Ws(2:end,:)];
