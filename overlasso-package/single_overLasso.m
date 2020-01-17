function res = single_overLasso(data,tridx,tsidx,lambdas, Winit, useAS, gweight, loss)
unshrink = 0;
options.display = 1000;
options.maxit=2e4;
options.tol = 1e-5;
options.bias = 0;
% Duplication step
edata = expandData(data);
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
np = sum(trdata.Y == 1);
nn = sum(trdata.Y == -1);
n = length(trdata.Y);
Cp = nn/n;
Cn = np/n;                     
f = @(b,w,s,data,atZero) blrOracle(b,w,s,data,atZero,Cp,Cn);
fH = @(b,w,s,data) blrH(b,w,s,data,Cp,Cn);
fGL = @(b,w,data) blrGL(b,w,data,Cp,Cn);
ftest = 'all'; % Return accuracy, balanced accuracy and AUC.

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

res.auc = -ones(1,length(lambdas));
res.acc = -ones(1,length(lambdas));
res.bacc = -ones(1,length(lambdas));
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
        x = tsdata.X;
        y = tsdata.Y;
        posidx = find(y>0);
        negidx = find(y<0);
        
        ypred = x*Ws(:,i);
        res.pred(:,i) = x*Ws(:,i);
        err.auc = auc(ypred,y);
        err.acc = sum(y ~= sign(ypred))/length(y);
        err.bacc = (sum(sign(ypred(posidx)) ~= 1)/max(1,length(posidx)) ...
        + sum(sign(ypred(negidx)) ~= -1)/max(1,length(negidx)))/2;
    
        res.auc(i) = err.auc;
        res.acc(i) = err.acc;
        res.bacc(i) = err.bacc;
    end
end

res.perf = res.bacc;
res.AS = ASarr;
res.completeAS = completeASarr;
res.Ws = Ws;
res.oWs = [Ws(1,:); edata.exp2orig*Ws(2:end,:)];
