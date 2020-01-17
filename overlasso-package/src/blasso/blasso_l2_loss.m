function msr = blasso_l2_loss(parameters)
b = parameters{1};
y = parameters{2};
x = parameters{3};

n_obs   = size(x,1);
n_betas = size(b,1);

for i = 1:n_betas
	res				= y-x*b(i,:)';
	msr(i,1)	= res'*res/n_obs;
end;