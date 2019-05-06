clear variables;
eps_step = 0.001;

z = load('matlab_diabetes_data.txt');

x = z(:,1:end-1);
y = z(:,end);

blasso_res = blassol2(y, x, eps_step);
fsf_res    = discrete_fsf(y, x, eps_step);

% Finally, compare the results:
figure(1);
plot(blasso_res.L1_nbetas, blasso_res.nbetas);
title('BLASSO path')

figure(2);
plot(fsf_res.L1_nbetas, fsf_res.nbetas);
title('FSF path');
