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


function [BW AS completeAS nextGrad] = blocktsengAS(f, fH, fGL, BW0, as, data, lambda, d, options)

% Active set version of Tseng et al. algorithm. Just calls blocktseng on a
% restricted set of variables, then check if the solution is a global
% optimum, and if not, adds the groups whose gradient norm is larger than
% lambda.

NADD = 100; % How many constraints do we add (at most) in the active set at each step?

b = BW0(1);
W = BW0(2:end);

groups = data.groups;

gridx = 1:length(groups);

AS = as;

% sidx must contain the indices of the variables from all the groups in the
% active set.
sidx = [];
for g=AS
    sidx = [sidx data.groups{g}];
end
globalSol = 0;
first = 1;
run = 0;

% Iterate while no global solution is found and the number of selected
% variables is less than maxFeat
while ~globalSol && (length(sidx) <= options.maxFeat)
    %fprintf('[blocktsengAS] Running optimization with %d groups in the active set\n',length(AS));
    % Build a data set restricted to the variables in AS
    subdata.X = data.X(:,sidx);
    subdata.Y = data.Y;
    subdata.groups = cell(1,length(AS));
    c = 0;
    for g=AS
        c = c+1;
        subdata.groups{c} = find(ismember(sidx,data.groups{g}));
    end
    % Optimize group lasso on the restricted dataset.
    sbw = blocktseng(f, fH, [b; W(sidx)], subdata, lambda, d(AS), options);
    b = sbw(1);
    w = sbw(2:end);
    W = zeros(size(data.X,2),1);
    W(sidx) = w;
    ngc = zeros(length(gridx),1);
    % Recompute the loss gradient at the new W.
    %fprintf('[blocktsengAS] Recomputing the full gradient.\n');
    fullGL = feval(fGL, b, W, data);
    %fprintf('[blocktsengAS] Full gradient recomputed, building the new active set.\n');
    for s=gridx
        ngc(s) = norm(fullGL(data.groups{s}));
    end
    if isempty(AS)
        if first % If active set gets empty one time (should be only at initialization), re-start with the highest gradient.
            [dummy AS] = max(ngc);
            first = 0;
        else
            break;
        end
    end
    % Add to the active set the groups whose gradient norm is more than
    % lambda
    lagr = max(ngc(AS)); % This is equal to lambda in theory. In practice, we use min(lambda,lagr).
    nasidx = find(~ismember(1:length(gridx),AS) &  (ngc' > min(lambda,lagr)));
    if any(nasidx)
        tmp = ngc(nasidx);
        [ngcmax ngcargmax] = sort(tmp,'descend');%max(tmp);
        ngcargmax = ngcargmax(1:min(NADD,length(nasidx)));
        AS = [nasidx(ngcargmax) AS];
        sidx = [];
        for g=AS
            sidx = [sidx data.groups{g}];            
        end
    else
        globalSol = 1;
        nextGrad = max(ngc(~ismember(1:length(gridx),AS)));
    end    
	run = run + 1;
end
% Find groups which could be active in other argmin
if ~isempty(AS)
    completeAS = gridx(ngc >= lambda-max(abs(lambda - ngc(AS))));
else
    completeAS = [];
end
BW = [b; W];
ASP = [];
for s=gridx
    if ismember(s,AS) && norm(W(data.groups{s}))
          ASP = [ASP s];
    end
end
% Remove from AS the groups that vanished during optimization
AS = ASP;

