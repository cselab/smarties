clc; clear

 titles = { 'ACKLEY', 'DIXON PRICE', 'GRIEWANK', 'LEVY', 'PERM', 'PERM0', ...
            'RASTRIGIN', 'ROSENBROCK', 'ROTATED HYPER ELLIPSOID', ...
            'SCHWEFEL', 'SPHERE', 'STYBLINSKI TANG', 'SUM OF POWER', 'SUM OF SQUARES', 'ZAKHAROV'};
        

s = dir('simulation_*');

D = importdata(['./' s.name '/cma_perf_00.dat'],' ',1);

d = D.data; clear D;

TOL = 1e-2;

fun_ids = sort(unique(d(:,1)));
n_fun_ids = length(fun_ids);

dims = sort(unique(d(:,2)));
n_dims = length(dims);

%% 
mean_steps = zeros(n_fun_ids,n_dims);
std_steps  = zeros(n_fun_ids,n_dims);
success_prob    = zeros(n_fun_ids,n_dims);
N  = zeros(n_fun_ids,n_dims);

mean_steps_tot = zeros(1,n_dims);
std_steps_tot  = zeros(1,n_dims);
success_prob_tot    = zeros(1,n_dims);
N_tot  = zeros(1,n_dims);


for j = 1:n_dims
    for i = 1:n_fun_ids
    
        index = d(:,1)==fun_ids(i) & d(:,2)==dims(j);
        tmp = d( index ,3);
        
        N(i,j) = sum(index);
        
        mean_steps(i,j) = mean(tmp);
        std_steps(i,j)  = std(tmp);
        
        success_prob(i,j) = sum( d(index,4) < TOL ) / N(i,j);
        
    end
    
    index = d(:,2)==dims(j);
    tmp = d( index ,3);

    N_tot(j) = sum(index);

    mean_steps_tot(j) = mean(tmp);
    std_steps_tot(j)  = std(tmp);

    success_prob_tot(j) = sum( d(index,4) < TOL ) / N_tot(j);

end





for i = 1:n_fun_ids
    
    figure(i)
    
    title(titles(fun_ids(i)+1));
    
    yyaxis left
    
    errorbar( dims, mean_steps(i,:), std_steps(i,:) );
    ylabel('Number of iterations to convergence')
    hold on

    yyaxis right
    plot(dims, success_prob(i,:))
    ylabel(['P of finding optimum (up to ' sprintf('%.1e',TOL) ')'])

    xlabel('Function dimensionality')

    grid on
    ax=gca;
    ax.FontSize = 15;
    
end


figure(i+1)
    
title('TOTAL');

yyaxis left

errorbar( dims, mean_steps_tot, std_steps_tot );
ylabel('Number of iterations to convergence')
hold on

yyaxis right
plot(dims, success_prob_tot)
ylabel(['P of finding optimum (up to ' sprintf('%.1e',TOL) ')'])

xlabel('Function dimensionality')

grid on
ax=gca;
ax.FontSize = 15;





