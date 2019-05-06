function [cvRes_lasso,cvRes_overlap, data pathways splitsstr] = mainBCPath()
% Breast Cancer data with pathway-based penalty
addpath(genpath('blasso'))
addpath(genpath('overlib'))
% parameters
% Which preprocessing apply to X
% Number of folds
nfolds = [5 5];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% getData %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('vant','X','Y','entrez')
Y = Y(:,2);
supersplits = cvsplitBal(Y,nfolds(1),fix(100*sum(clock)));
for i=1:nfolds(1)
    subsplits{i} = cvsplitBal(Y(supersplits{i}.train), ...
    nfolds(2),fix(100*sum(clock)));
end        
data.X = X;
data.Y = Y;
data.entrez = entrez;
splits.splits = supersplits;
splits.subsplits = subsplits;
splitsstr = splits;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PreProcess %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[mainBCPath] Data formated\n');
pathways=load('pathways.txt');
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
lambdas = fliplr(2.^(-12:0.1:0));
tic
fprintf('[mainBCGr] Starting lasso\n');
data.groups = num2cell(1:size(data.X,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cvOverLasso %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvRes_lasso = cell(1,nfolds);

splits = splitsstr.splits;
subsplits = splitsstr.subsplits;
nfolds = length(splits);
nsfolds = length(subsplits{1});


gweight = 0;
useAS = 1;
for ifold=(1:nfolds)
    fprintf('[cvOverLasso] Learning on fold %i\n',ifold);   
    tridx = splits{ifold}.train;
    testidx = splits{ifold}.test;
    fdata = data;
    %Split data
    trdata.X = fdata.X(tridx,:);
    trdata.Y = fdata.Y(tridx);
    trdata.entrez = fdata.entrez;
    trdata.groups = fdata.groups;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % [pdata kidx,kgroups,kgroupidx] = process(data,type)
    p = size(trdata.X,2);
    cvR = zeros(1,p);
    for i = 1:p
        C = corrcoef(trdata.X(:,i),trdata.Y);
        cvR(i) = C(1,2);
    end
    [ts,idx]=sort(abs(cvR),'descend');
    kidx = idx(1:500);
    pdata.Y = trdata.Y;
    pdata.X = trdata.X(:,kidx);
    pdata.entrez = trdata.entrez(kidx);
    
    % Restrict groups to remaining variables, and convert variable numbers.
    groups = data.groups;
    c = 0;
    for g =1:length(groups)
        rv = groups{g}(ismember(groups{g},kidx));
        if ~isempty(rv)
            c = c+1;
            tmp = zeros(1,length(rv));
            i = 0;
            for v=rv
                i = i+1;
                tmp(i) = find(kidx == v);
            end
            pgroups{c} = tmp;
            kgroups{c} = rv;
            kgroupidx(c) = g;
        end     
    end
    pdata.groups = pgroups;
    trdata = pdata;
    fprintf('[process] Pre-process over. Kept %d(/%d) variables', ...
    length(kidx),size(data.X,2));
    fprintf(' and %d(/%d) groups\n', length(pgroups),length(groups));
        
    fdata.X = fdata.X(:,kidx);
    fdata.entrez = trdata.entrez;
    fdata.groups = trdata.groups;

    cvRes_lasso{ifold}.lambdas = lambdas;
    cvRes_lasso{ifold}.kidx = kidx;
    cvRes_lasso{ifold}.kgroups = kgroups;
    cvRes_lasso{ifold}.kgroupidx = kgroupidx;
    cvRes_lasso{ifold}.groups = fdata.groups;
       
    sauc = zeros(1,length(lambdas));
    sacc = zeros(1,length(lambdas));
    sbacc = zeros(1,length(lambdas));
    % Internal CV to select lambda
    lres = cell(1,nsfolds);
    for isfold=(1:nsfolds)
        fprintf('[cvOverLasso] In internal CV, learning on subfold %i\n',isfold);
        lres{isfold} = single_overLasso(fdata,subsplits{ifold}{isfold}.train, ...
        subsplits{ifold}{isfold}.test,lambdas,[],useAS,gweight,'blr');
        sauc = sauc+(lres{isfold}.auc/nfolds);
        sacc = sacc+(lres{isfold}.acc/nfolds);
        sbacc = sbacc+(lres{isfold}.bacc/nfolds);
    end
    [dummy lstar] = min(sbacc);
    cvRes_lasso{ifold}.sbacc = sbacc;
    cvR = single_overLasso(fdata,tridx,testidx,lambdas,[],useAS,gweight,'blr');
    cvRes_lasso{ifold}.AS = cvR.AS;
    cvRes_lasso{ifold}.completeAS = cvR.completeAS;
    cvRes_lasso{ifold}.lstar = lambdas(lstar);
    cvRes_lasso{ifold}.auc = cvR.auc;
    cvRes_lasso{ifold}.acc = cvR.acc;
    cvRes_lasso{ifold}.bacc = cvR.bacc;
    cvRes_lasso{ifold}.perf = cvR.bacc(lstar);
    
    cvRes_lasso{ifold}.pred = cvR.pred;
    cvRes_lasso{ifold}.Ws = cvR.Ws;
    cvRes_lasso{ifold}.oWs = cvR.oWs;    
    cvRes_lasso{ifold}.nextGrad = cvR.nextGrad;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Groups are the pathways
fprintf('[mainBCGr] Starting structure lasso on pathways\n');
data.groups = pgroups;
% data,splitsstr,lambdas,proctype,varargin
splits = splitsstr.splits;
subsplits = splitsstr.subsplits;
nfolds = length(splits);
nsfolds = length(subsplits{1});
cvRes_overlap = cell(1,nfolds);
gweight = 0; % Should we reweight the group norms in the penalty?
useAS = 1;
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
    % [trdata kidx kgroups kgroupidx] = process(trdata,proctype);
    p = size(trdata.X,2);
    cvR = zeros(1,p);
    for i = 1:p
        C = corrcoef(trdata.X(:,i),trdata.Y);
        cvR(i) = C(1,2);
    end
    [ts,idx]=sort(abs(cvR),'descend');
    kidx = idx(1:500);
    pdata.Y = trdata.Y;
    pdata.X = trdata.X(:,kidx);
    pdata.entrez = trdata.entrez(kidx);
    
    % Restrict groups to remaining variables, and convert variable numbers.
    groups = data.groups;
    c = 0;
    for g =1:length(groups)
        rv = groups{g}(ismember(groups{g},kidx));
        if ~isempty(rv)
            c = c+1;
            tmp = zeros(1,length(rv));
            i = 0;
            for v=rv
                i = i+1;
                tmp(i) = find(kidx == v);
            end
            pgroups{c} = tmp;
            kgroups{c} = rv;
            kgroupidx(c) = g;
        end     
    end

    pdata.groups = pgroups;
    trdata = pdata;
    fprintf('[process] Pre-process over. Kept %d(/%d) variables', ...
    length(kidx),size(data.X,2));
    fprintf(' and %d(/%d) groups\n', length(pgroups),length(groups));
    
    
    fdata.X = fdata.X(:,kidx);
    fdata.entrez = trdata.entrez;
    fdata.groups = trdata.groups;

    cvRes_overlap{ifold}.lambdas = lambdas;
    cvRes_overlap{ifold}.kidx = kidx;
    cvRes_overlap{ifold}.kgroups = kgroups;
    cvRes_overlap{ifold}.kgroupidx = kgroupidx;
    cvRes_overlap{ifold}.groups = fdata.groups;
       
    sauc = zeros(1,length(lambdas));
    sacc = zeros(1,length(lambdas));
    sbacc = zeros(1,length(lambdas));
    % Internal CV to select lambda
    lres = cell(1,nsfolds);
    for isfold=(1:nsfolds)
        fprintf('[cvOverLasso] In internal CV, learning on subfold %i\n',isfold);
        lres{isfold} = single_overLasso(fdata,subsplits{ifold}{isfold}.train, ...
        subsplits{ifold}{isfold}.test,lambdas,[],useAS,gweight,'blr');
        sauc = sauc+(lres{isfold}.auc/nfolds);
        sacc = sacc+(lres{isfold}.acc/nfolds);
        sbacc = sbacc+(lres{isfold}.bacc/nfolds);
    end
    [dummy lstar] = min(sbacc);
    cvRes_overlap{ifold}.sbacc = sbacc;
    cvR = single_overLasso(fdata,tridx,testidx,lambdas,[],useAS,gweight,'blr');
    cvRes_overlap{ifold}.AS = cvR.AS;
    cvRes_overlap{ifold}.completeAS = cvR.completeAS;
    cvRes_overlap{ifold}.lstar = lambdas(lstar);
    
    cvRes_overlap{ifold}.auc = cvR.auc;
    cvRes_overlap{ifold}.acc = cvR.acc;
    cvRes_overlap{ifold}.bacc = cvR.bacc;
    cvRes_overlap{ifold}.perf = cvR.bacc(lstar);
    cvRes_overlap{ifold}.pred = cvR.pred;
    cvRes_overlap{ifold}.Ws = cvR.Ws;
    cvRes_overlap{ifold}.oWs = cvR.oWs;    
    cvRes_overlap{ifold}.nextGrad = cvR.nextGrad;
end
str = sprintf('/network/rit/lab/ceashpc/bz383376/data/icml19/breast_cancer/output/test.mat');
save(str, 'cvRes_lasso', 'cvRes_overlap', 'data', 'pathways', 'splitsstr')
