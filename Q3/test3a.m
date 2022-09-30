my1=1;
my2=-1;
sigma1=1;
sigma2=4;
beta=5;

sum= inv(inv([sigma1 0;0 sigma2])+[1 -1]'*(1/beta)*[1 -1]);
my = sum*(inv([sigma1 0; 0 sigma2])*[my1; my2]+[1 -1]'*(1/beta).*(3))