function threeDoF_try3()

clear
close all
clc

yVecNot = [-10, 0, 0]; % [u0, v0, omega0]
tRange = [0,2];

tt = linspace(tRange(1), tRange(2),1000);
%theta = 2*pi/(tRange(2) - tRange(1));

thrustLeft = 20*ones(size(tt));
thrustRight = -20*ones(size(tt));

% Coord system seems to be messed up, using sign which gives expected
% result
torque = thrustRight - thrustLeft; % *0.5*width - do this in the derivComput

forceX = thrustLeft + thrustRight;
forceY = 0*ones(size(tt)); %sin(theta*tt) %.* randn(size(tt));

figure; plot(tt,forceX,tt,forceY); legend('forceX', 'forceY')

[tInteg, y] = ode15s(@(t,y) derivs(t,y, tt,forceX,forceY,torque), tt, yVecNot);
%[t, y] = ode45(@derivs, tRange, yVecNot);

figure
plot(tInteg, y(:,1), tInteg, y(:,2), tInteg, y(:,3))
legend('u', 'v', '\omega')

%Now integrate u, v to get the trajectory
[tTraj,traj] = ode45(@(tTraj, traj) speed(tTraj, traj, tInteg, y(:,1), y(:,2)), tInteg, [0;0]);

figure;
plot(traj(:,1), traj(:,2)); xlabel('x'); ylabel('y')

figure
plot(traj(:,2), traj(:,1)); xlabel('y'); ylabel('x')
hold on; plot(traj(1,2), traj(1,1), 'or','MarkerFacecolor','red'); plot(traj(end,2), traj(end,1), 'rx', 'MarkerSize', 10)
title(['u0=', num2str(yVecNot(1)), ', thrustL=',num2str(thrustLeft(1)),', thrustR=', num2str(thrustRight(1))],'FontSize',12)
quiver(traj(:,2), traj(:,1), forceY', forceX')
axis equal

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
function retDeriv = derivs(t, yVec, tt, forceX, forceY,torque)

t

% Parameters
m = 280;
Iz = 300;
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
Fy = interp1(tt,forceY,t);
Tau = interp1(tt,torque,t)*0.5*l;

u = yVec(1); v = yVec(2); r = yVec(3);

retDeriv = zeros(3,1);


% Simplified model
% retDeriv(1) = Fx + ( (m-YvDot)*v*r - Xu*u ) / (m-XuDot);
% retDeriv(2) = Fy + (-(m-XuDot)*u*r - Yv*v ) / (m-YvDot);
% retDeriv(3) = Tau + ( (YvDot-XuDot)*u*v - Nr*r ) / (Iz-NrDot);

% Full model
M = [   m-XuDot 0       0;
        0       m-YvDot -YrDot;
        0       -NvDot   Iz-NrDot];
    
C_rb = [0       0       -m*v;
        0       0       m*u;
        m*v     -m*u    0];

C_am = [0       0       YvDot*v + (YrDot + NvDot)*r/2;
        0       0       -XuDot*u;
        -(YvDot*v + (YrDot + NvDot)*r/2)       XuDot*u     0;];
    
C = C_rb + C_am;

D_l = [ Xu   0   0;
        0   Yv   Yr;
        0   Nv   Nr];
    
D_nl = zeros(size(D_l));
    
D = D_l + D_nl;

retDeriv = -inv(M)*(C + D)*[u; v; r] + [Fx; Fy; Tau];

end

function retVal = speed(t, y, tInteg, uu, vv)
    retVal = zeros(2,1);
    retVal(1) = interp1(tInteg, uu, t);
    retVal(2) = interp1(tInteg, vv, t);
end           