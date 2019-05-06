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

function cvRes = runMethod(data, splits, method, proctype, varargin)

% This function runs 'method' on 'data' splitted according to 'splits'. It
% just runs some preprocessings, loops on the folds and calls the
% optimization function.

nfolds = length(splits);
cvRes = cell(1,nfolds);

for ifold=(1:nfolds)
    fprintf('Learning on fold %i\n',ifold);

    tridx = splits{ifold}.train;
    testidx = splits{ifold}.test;

    % Will be processed for this fold
    fdata = data;

    % Split data
    trdata.X = fdata.X(tridx,:);
    trdata.Y = fdata.Y(tridx);
    trdata.entrez = fdata.entrez;
    trdata.groups = fdata.groups;

    kidx = 1:size(trdata.X,2);    
    kgroups = trdata.groups;
    kgroupidx = 1:length(trdata.groups);
    if ~strcmp(proctype,'plain')
        % Process based only on training data
        [trdata kidx kgroups kgroupidx] = process(trdata,proctype);
        fdata.X = fdata.X(:,kidx);
        fdata.entrez = trdata.entrez;
        fdata.groups = trdata.groups;
    end
    cvRes{ifold}.kidx = kidx;
    cvRes{ifold}.kgroupidx = kgroupidx;
    cvRes{ifold}.kgroups = kgroups;
    cvRes{ifold}.groups = trdata.groups;
        
    switch method
        case 'over'
            lambdas = fliplr(2.^(-7:0.2:3));
            loss = varargin{1};
            gweight = 0; % Should we reweight the group norms in the penalty?
            useAS = 1;
            fprintf('[runMethod] Entering over\n');
            res = overLasso(fdata,tridx,testidx,lambdas, [], useAS, gweight,loss);
            fprintf('[runMethod] Leaving over\n');
            
            cvRes{ifold}.lambdas = res.lambdas;
            if strcmp(loss,'blr')
                cvRes{ifold}.auc = res.auc;
                cvRes{ifold}.acc = res.acc;
                cvRes{ifold}.bacc = res.bacc;
                cvRes{ifold}.perf = res.bacc;
            else
                cvRes{ifold}.perf = res.perf;
            end
            cvRes{ifold}.pred = res.pred;
            cvRes{ifold}.AS = res.AS;
            cvRes{ifold}.completeAS = res.completeAS;
            cvRes{ifold}.Ws = res.Ws;
            cvRes{ifold}.oWs = res.oWs;
            cvRes{ifold}.nextGrad = res.nextGrad;
        case 'blasso'
            eps_step = 0.001;
            fprintf('[runMethod] Entering lasso\n');
            res = blassol2(trdata.Y, trdata.X, eps_step);
            fprintf('[runMethod] Leaving lasso\n');

            cvRes{ifold}.lambdas = res.lambdas;

            ll = length(res.lambdas);
            Ws = zeros(size(trdata.X,2)+1,ll);

            tsdata.X = [ones(length(testidx),1) fdata.X(testidx,:)];
            tsdata.Y = fdata.Y(testidx);

            for l=1:ll
                w = [res.intercepts(l) res.betas(l,:)];
                Ws(:,l) = w';
                [cvRes{ifold}.perf(l) cvRes{ifold}.ypred{l}] = test(tsdata,Ws(:,l),'l2');
            end
            cvRes{ifold}.Ws = Ws;
            cvRes{ifold}.oWs = Ws;
        otherwise
            error('Learning method not implemented');
    end
end
