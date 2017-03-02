% Create polar data
[r,t] = meshgrid(0:1/120:1,0:pi/120:(2*pi));

value = 10 * r.^4.*sin(4 * t);
% Convert to Cartesian
[x,y,z] = pol2cart(t,r,value);

figure(1);
h = surf(x,y,z);
set(h,'edgeColor', 'none')
colormap cool;
hold on;