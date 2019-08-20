function threeDoF()

clear
close all
clc

yVecNot = [4, 0, 0]; % [u0, v0, omega0]
tRange = [0,10];

tt = linspace(tRange(1), tRange(2),100);
theta = 2*pi/(tRange(2) - tRange(1));
forceX = cos(theta/4*tt);
forceY = sin(theta*tt) %.* randn(size(tt));
figure; plot(tt,forceX,tt,forceY); legend('forceX', 'forceY')

[tInteg, y] = ode15s(@(t,y) derivs(t,y, tt,forceX,forceY), tt, yVecNot);
%[t, y] = ode45(@derivs, tRange, yVecNot);

figure
plot(tInteg, y(:,1), tInteg, y(:,2), tInteg, y(:,3))
legend('u', 'v', '\omega')

%Now integrate u, v to get the trajectory
[tTraj,traj] = ode45(@(tTraj, traj) speed(tTraj, traj, tInteg, y(:,1), y(:,2)), tInteg, [0;0]);

figure
plot(traj(:,2), traj(:,1)); xlabel('y'); ylabel('x')
hold on; plot(traj(1,2), traj(1,1), 'bo'); plot(traj(end,2), traj(end,1), 'bx')
quiver(traj(:,2), traj(:,1), forceY', forceX')

% % Make movie
% xx = traj(:,1);
% yy = traj(:,2);
% 
% vidObj = VideoWriter('trajectory.mp4','MPEG-4');
% vidObj.FrameRate = 15;
% open(vidObj);
% figure;
% for ind = 1:length(xx)
%     
%     clf; hold on;
%     plot(yy(1), xx(1), 'bo'); plot(yy(end), xx(end), 'bx');
%     plot(yy(1:ind), xx(1:ind));
%     quiver(yy(1:ind), xx(1:ind), forceY(1:ind)', forceX(1:ind)');
%     title(['time = ',num2str(tt(ind))]);
%     xlabel('y'); ylabel('x')
%     frame = getframe(gcf);
%     writeVideo(vidObj, frame);
%     
% end
% 
% close(vidObj);

end



% derivative
function retDeriv = derivs(t, yVec, tt, forceX, forceY)

t

% Parameters
m = 280;
Izz = 300;
l = 1.83;
Xu = 86.45;
Xuu = 0;
Yv = 300;
Nr = 500;
Nv = -250;
Yr = -80;
Nu = 20;
XuDot = -30;
YvDot = -40;
NrDot = -90;
NvDot = -50;
YrDot = -50;

% Can make these functions of time, to put in a random number as well as
% motor control
Fx = interp1(tt,forceX,t);
stdDeviation = 0.0;
%Fy = randn(1)*stdDeviation; %Random buffeting from waves
Fy = interp1(tt,forceY,t);
Tau = 0;

u = yVec(1); v = yVec(2); r = yVec(3);
% For now, use decoupled. Later in C++ code use linearization via most
% recent available value
rDot = 0;
vDot = 0;

retDeriv = zeros(3,1);
retDeriv(1) = (Fx + r*(YvDot*v + m*v - r*(NvDot + YrDot)/2) - u*(Xu + Xuu*u)) / (m-XuDot);
retDeriv(2) = (Fy - r*(XuDot*u + m*u) - Yv*v + YrDot*rDot - Yr*r) / (m-YvDot);
retDeriv(3) = (Tau + NvDot*vDot - Nv*v - u*(YvDot*v + m*v + r*(NvDot+YrDot)/2) + Nr*r + v*(XuDot*u + m*u)) / (Izz-NrDot);

end

function retVal = speed(t, y, tInteg, uu, vv)
    retVal = zeros(2,1);
    retVal(1) = interp1(tInteg, uu, t);
    retVal(2) = interp1(tInteg, vv, t);
end

%

% Mmatrix = [ m - XuDot, 0, 0;
%             0, m - YvDot, -YrDot;
%             0, -NvDot, Izz - NrDot];
% 
% % The damping matrix depends on velocity vector. Linearize by plugging in
% % current timestep values.
% Cmatrix = [ 0, 0, -(m-YvDot)*v + r*(YrDot + NvDot)/2;
%             0, 0, (m - XuDot)*u;
%             (m-YvDot)*v - r*(YrDot + NvDot)/2, -(m - XuDot)*u, 0];
%         
% Dmatrix = -[    Xuu*u + Xu, 0, 0;
%                 0, Yv, Yr;
%                 0, Nv, Nr];
%           