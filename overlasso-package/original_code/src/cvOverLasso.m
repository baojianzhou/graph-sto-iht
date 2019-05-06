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

function cvRes = cvOverLasso(data,splitsstr,lambdas,proctype,varargin)

splits = splitsstr.splits;
subsplits = splitsstr.subsplits;
nfolds = length(splits);
nsfolds = length(subsplits{1});
cvRes = cell(1,nfolds);

loss = varargin{1};
gweight = 0; % Should we reweight the group norms in the penalty?
useAS = 1;

%nboot = 50; % Number of bootstrap runs;

for ifold=(1:nfolds)
    fprintf('[cvOverLasso] Learning on fold %i\n',ifold);   
    tridx = splits{ifold}.train;
    testidx = splits{ifold}.test;
    % Will be processed for this fold
    fdata = data;
    %Split data
    trdata.X = fdata.X(tridx,:);
    trdata.Y = fdata.Y(tridx);
    trdata.entrez = fdata.entrez;
    trdata.groups = fdata.groups;
    % Process based only on training data
    [trdata kidx kgroups kgroupidx] = process(trdata,proctype);
    fdata.X = fdata.X(:,kidx);
    fdata.entrez = trdata.entrez;
    fdata.groups = trdata.groups;

    cvRes{ifold}.lambdas = lambdas;
    cvRes{ifold}.kidx = kidx;
    cvRes{ifold}.kgroups = kgroups;
    cvRes{ifold}.kgroupidx = kgroupidx;
    cvRes{ifold}.groups = fdata.groups;
       
    
    if strcmp(loss,'blr')
        sauc = zeros(1,length(lambdas));
        sacc = zeros(1,length(lambdas));
        sbacc = zeros(1,length(lambdas));
    else
        sperfs = zeros(1,length(lambdas));
    end
    
    % Internal CV to select lambda
    lres = cell(1,nsfolds);
    for isfold=(1:nsfolds)
        fprintf('[cvOverLasso] In internal CV, learning on subfold %i\n',isfold);
        lres{isfold} = overLasso(fdata,subsplits{ifold}{isfold}.train,subsplits{ifold}{isfold}.test,lambdas,[],useAS,gweight,loss);
        %lres{isfold} = boverLasso(fdata,subsplits{ifold}{isfold}.train,subsplits{ifold}{isfold}.test,lambdas,[],useAS,gweight,loss,nboot);
        if strcmp(loss,'blr')
            sauc = sauc+(lres{isfold}.auc/nfolds);
            sacc = sacc+(lres{isfold}.acc/nfolds);
            sbacc = sbacc+(lres{isfold}.bacc/nfolds);
        else
            sperfs = sperfs+(lres{isfold}.perf/nfolds);
        end        
    end
    if strcmp(loss,'blr')
        [dummy lstar] = min(sbacc);
        cvRes{ifold}.sbacc = sbacc;
    else
        [dummy lstar] = min(sperfs);
    end
    
    
%     res.auc = -ones(1,length(lambdas));
%     res.acc = -ones(1,length(lambdas));
%     res.bacc = -ones(1,length(lambdas));
%     res.pred = zeros(size(trdata.X,1),length(lambdas));
%     res.nextGrad = -ones(1,length(lambdas));    
%     res.Ws = zeros(size(trdata.X,2)+1,length(lambdas));
%     res.AS = cell(1,length(lambdas));
%     res.completeAS = cell(1,length(lambdas));
%     i = 0;
%     for lambda=lambdas
%         i = i+1;
%         AS = [];
%         for isfold=(1:nsfolds)
%             AS = union(AS,lres{isfold}.completeAS{i});
%         end
%         llres = lr(fdata,tridx,testidx,lambda,AS,'all');
%         res.auc(i) = llres.auc;
%         res.acc(i) = llres.acc;
%         res.bacc(i) = llres.bacc;
%         res.pred(:,i) = llres.bacc;
%         %res.Ws(:,i) = llres.Ws;
%         %res.oWs(:,i) = llres.Ws;
%         res.AS{i} = llres.AS;
%         res.completeAS{i} = llres.completeAS;
%     end
    
    % res = overLasso(fdata,tridx,testidx,lambdas(lstar),[],useAS,gweight,loss);
     res = overLasso(fdata,tridx,testidx,lambdas,[],useAS,gweight,loss);
    %res = boverLasso(fdata,tridx,testidx,lambdas,[],useAS,gweight,loss,nboot);
    %res = pooledUnshrink(fdata,tridx,testidx,lambdas,[], lres, gweight, loss);

    % Selected groups in found argmin and in union(argmin)
    cvRes{ifold}.AS = res.AS;
    cvRes{ifold}.completeAS = res.completeAS;
    
    cvRes{ifold}.lstar = lambdas(lstar);
    if strcmp(loss,'blr')
        cvRes{ifold}.auc = res.auc;
        cvRes{ifold}.acc = res.acc;
        cvRes{ifold}.bacc = res.bacc;
        cvRes{ifold}.perf = res.bacc(lstar);
    else
        cvRes{ifold}.perf = res.perf;
    end
    cvRes{ifold}.pred = res.pred;
    cvRes{ifold}.Ws = res.Ws;
    cvRes{ifold}.oWs = res.oWs;    
    cvRes{ifold}.nextGrad = res.nextGrad;
end