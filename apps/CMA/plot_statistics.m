load stats.txt;
str1 = {'epoch','avg mse','avg rel err','avg Q','min Q','max Q','errWeights','errWeights','errWeights','N','steps','dT'};

ind=[4,5,6];

for k=ind
    plot(stats(:,1),stats(:,k),'LineWidth',2);
    hold on
end
legend(str1(ind),'Location','best')
ax=gca;
ax.FontSize=15;
axis tight; grid on

%% -------------------------------------------------------

figure(); clf
load master_rewards.dat;
str2 = {'Iter','Mean reward','variance'};
plot(master_rewards(:,1),master_rewards(:,2),'LineWidth',2); hold on
plot(master_rewards(:,1),master_rewards(:,3),'LineWidth',2);

legend('mean reward','std','Location','best')
ax=gca;
ax.FontSize=15;
axis tight; grid on


%% -------------------------------------------------------
figure(); clf
tmp = load('obs_agent_0.dat');
fl = tmp(:,2);
r = tmp(:,end);


ind = find(fl==2);
N = length(ind);

sm = zeros(N,1);
sm(1) = sum(r(1:ind(1)));
for i=2:N
    sm(i) = sum(r(ind(i-1)+1:ind(i)));
end

plot(cumsum(sm)./(1:N)','LineWidth',2);
ax=gca;
ax.FontSize=15;
axis tight; grid on
    
    
    
    
