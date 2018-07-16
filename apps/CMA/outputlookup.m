% function visualize_CMARL
close all
colors = [0 .447 .741; .85 .325 .098; .929 .694 .125; .494 .184 .556;
            .466 .674 .188; .301 .745 .933; .635 .078 .184];

s = dir('simulation_*');
D = importdata(['./' s.name '/cma_perf_00.dat'],' ',1);

D = D.data;
plot_data(D, 0, colors(1,:), '.-')
nD = size(D,1);

% 
% %D1=importdata(['/Volumes/Apps/smarties/runs/CMA_naf_lambda_smallpenal_2'...
% D1=importdata(['/Users/novatig/Desktop/euler/smarties/apps/CMA/cma_perf_RL.dat'],' ',1);
% D = D1.data;
% D = D(end-nD:end,:);
% % D1=importdata('/Volumes/Apps/smarties/runs/CMA_naf_prod/simulation_1_259/cma_perf_00.dat',' ',1);
% % D = [D; D1.data];
% % D1=importdata('/Volumes/Apps/smarties/runs/CMA_naf_prod/simulation_1_260/cma_perf_00.dat',' ',1);
% % D = [D; D1.data];
% %D = D(floor(length(D)/2):end,:);
% plot_data(D, 1, colors(2,:), '-')
% % end



function plot_data(D, bHold, color, line)
    maxdim = max(D(:,1));
    ndata = length(D);
    countfun     = zeros(15,maxdim);
    sum1step     = zeros(15,maxdim);
    sum2step     = zeros(15,maxdim);
    sum1conv     = zeros(15,maxdim);
    countfun_dim = zeros(1, maxdim);
    sum1step_dim = zeros(1, maxdim);
    sum2step_dim = zeros(1, maxdim);
    sum1conv_dim = zeros(1, maxdim);

    titles={'Average', 'ACKLEY', 'DIXON PRICE', 'GRIEWANK', 'LEVY', 'PERM', 'PERM0', ...
            'RASTRIGIN', 'ROSENBROCK', 'ROTATED HYPER ELLIPSOID', ...
            'SCHWEFEL', 'SPHERE', 'STYBLINSKI TANG', 'SUM OF POWER', 'SUM OF SQUARES', 'ZAKHAROV'};

    %func_dim, info[0], step, final_dist, ffinal
    for l = 1:ndata
        ndim   = D(l,1);
        funcID = D(l,2)+1;
        nsteps = D(l,3);
        opdist = D(l,4);

        countfun(funcID, ndim) = countfun(funcID, ndim) + 1;
        delta = nsteps - sum1step(funcID, ndim);
        sum1step(funcID, ndim) = sum1step(funcID, ndim) + delta/countfun(funcID, ndim);
        delta2 = nsteps - sum1step(funcID, ndim);
        sum2step(funcID, ndim) = sum2step(funcID, ndim) + delta*delta2;

        countfun_dim(1, ndim) = countfun_dim(1, ndim) + 1;
        delta = nsteps - sum1step_dim(1, ndim);
        sum1step_dim(1, ndim) = sum1step_dim(1, ndim) + delta/countfun_dim(1, ndim);
        delta2 = nsteps - sum1step_dim(1, ndim);
        sum2step_dim(1, ndim) = sum2step_dim(1, ndim) + delta*delta2;

        TOL = 1e-2;
        if opdist < TOL
            sum1conv(funcID, ndim) = sum1conv(funcID, ndim) + 1;
            sum1conv_dim(1, ndim) = sum1conv_dim(1, ndim) + 1;
        end
    end

    % avg_iter = [sum1step_dim./countfun_dim; 
    %                 sum1step./countfun];
      avg_iter = [sum1step_dim; sum1step];      
      std_iter = [sqrt(sum2step_dim./(countfun_dim-1)); sqrt(sum2step./(countfun-1))];
    % std_iter = [sqrt((sum2step_dim-sum1step_dim.*sum1step_dim./countfun_dim)./countfun_dim);
    %             sqrt((sum2step-sum1step.*sum1step./countfun)./countfun)];

    conv_prob = [sum1conv_dim./countfun_dim; sum1conv./countfun];
    dims = [2 3 4 5 6 7 8 9 10];

    for i = 10
        figure(i)
        if bHold == 1 hold on
        else hold off
        end
        title(titles(i));

        yyaxis left
        errorbar(dims,avg_iter(i,2:10),std_iter(i,2:10),line)
        ylabel('Number of iterations to convergence')
        hold on

        yyaxis right
        plot(dims, conv_prob(i,[2:10]),line)
        ylabel(['P of finding optimum (up to ' sprintf('%.1e',TOL) ')'])

        xlabel('Function dimensionality')

        grid on

        ax=gca;
        ax.FontSize = 15;



    end
end


