clear;
close all;
clc;

physical_constants;

%% =========================================================
% PARAMETRY
% =========================================================

f0 = 435e6;
fc = 200e6;

c0 = C0;
lambda0 = c0/f0;

% Dipol półfalowy (praktyczna długość)
L_dipole = 0.47 * lambda0;

wire_radius = 2e-3;      % 2 mm
gap = 4e-3;              % 4 mm

SimBox = [2 2 2] * lambda0;

%% =========================================================
% FDTD
% =========================================================

NrTS = 60000;
EndCriteria = 1e-5;

FDTD = InitFDTD( ...
    'NrTS', NrTS, ...
    'EndCriteria', EndCriteria);

FDTD = SetGaussExcite(FDTD, f0, fc);

BC = {'PML_8' 'PML_8' 'PML_8' ...
      'PML_8' 'PML_8' 'PML_8'};

FDTD = SetBoundaryCond(FDTD, BC);

%% =========================================================
% CSX
% =========================================================

CSX = InitCSX();

%% =========================================================
% METAL
% =========================================================

CSX = AddMetal(CSX, 'PEC');

%% =========================================================
% DIPOL
% =========================================================

% górne ramię
start = [0 0 gap/2];
stop  = [0 0 L_dipole/2];

CSX = AddCylinder(CSX, 'PEC', 10, ...
    start, stop, wire_radius);

% dolne ramię
start = [0 0 -gap/2];
stop  = [0 0 -L_dipole/2];

CSX = AddCylinder(CSX, 'PEC', 10, ...
    start, stop, wire_radius);

%% =========================================================
% PORT
% =========================================================

port_start = [0 0 -gap/2];
port_stop  = [0 0  gap/2];

[CSX, port] = AddLumpedPort( ...
    CSX, ...
    5, ...
    1, ...
    50, ...
    port_start, ...
    port_stop, ...
    [0 0 1], ...
    true);

%% =========================================================
% MESH
% =========================================================

max_res = lambda0/120;

mesh.x = [
    -SimBox(1)/2 ...
    -0.03 ...
    -0.01 ...
    -0.005 ...
    -wire_radius ...
     0 ...
     wire_radius ...
     0.005 ...
     0.01 ...
     0.03 ...
     SimBox(1)/2
];

mesh.y = mesh.x;

mesh.z = [
    -SimBox(3)/2 ...
    -L_dipole/2 ...
    -0.01 ...
    -gap ...
    -gap/2 ...
     0 ...
     gap/2 ...
     gap ...
     0.01 ...
     L_dipole/2 ...
     SimBox(3)/2
];

mesh.x = SmoothMeshLines(mesh.x, max_res, 1.3);
mesh.y = SmoothMeshLines(mesh.y, max_res, 1.3);
mesh.z = SmoothMeshLines(mesh.z, max_res, 1.3);

CSX = DefineRectGrid(CSX, 1.0, mesh);

%% =========================================================
% NF2FF
% =========================================================

%nf2ff = CreateNF2FFBox(CSX, 'nf2ff');

%% =========================================================
% DUMP POLA E
% =========================================================

dump_z = 0;

start = [-1 -1 dump_z];
stop  = [ 1  1 dump_z];

CSX = AddDump(CSX, ...
    'Et', ...
    'DumpType', 10, ...
    'FileType', 1, ...
    'Frequency', f0);

CSX = AddBox(CSX, 'Et', 0, start, stop);

%% =========================================================
% ZAPIS
% =========================================================

Sim_Path = 'dipole_435MHz';
Sim_CSX = 'dipole.xml';

%rmdir(Sim_Path, 's');
mkdir(Sim_Path);

WriteOpenEMS( ...
    [Sim_Path '/' Sim_CSX], ...
    FDTD, ...
    CSX);

%% =========================================================
% PODGLĄD
% =========================================================

%CSXGeomPlot([Sim_Path '/' Sim_CSX]);

%% =========================================================
% RUN
% =========================================================

RunOpenEMS(Sim_Path, Sim_CSX);

%% =========================================================
% POSTPROCESSING
% =========================================================

freq = linspace(200e6, 700e6, 501);

port = calcPort(port, Sim_Path, freq);

s11 = port.uf.ref ./ port.uf.inc;

Zin = port.uf.tot ./ port.if.tot;
Pin = port.P_acc;

[~, idx] = max(port.P_acc);
f_res = freq(idx)
Pmax = port.P_acc(idx);
Ptar = 10
scale = sqrt(Ptar / Pmax);
%% =========================================================
% S11
% =========================================================

figure;

plot(freq/1e6, ...
    20*log10(abs(s11)), ...
    'LineWidth', 2);

grid on;

xlabel('Frequency [MHz]');
ylabel('S11 [dB]');

title('Dipole S11');


figure;

plot(freq/1e6, ...
    Pin, ...
    'LineWidth', 2);

grid on;

xlabel('Frequency [MHz]');
ylabel('Pin [dB]');

title('Dipole Pin');
%% =========================================================
% IMPEDANCJA
% =========================================================

figure;

plot(freq/1e6, real(Zin), 'LineWidth', 2);
hold on;

plot(freq/1e6, imag(Zin), 'LineWidth', 2);

grid on;

xlabel('Frequency [MHz]');
ylabel('Impedance [Ohm]');

legend('Real(Z)', 'Imag(Z)');

title('Input Impedance');

%% =========================================================
% FAR FIELD
% =========================================================

theta = -180:2:180;

%nf2ff_res = CalcNF2FF( ...
%    nf2ff, ...
%    Sim_Path, ...
%    f0, ...
%    theta, ...
%    90);

%figure;

%polarplot( ...
%    deg2rad(theta), ...
%    squeeze(nf2ff_res.E_norm));

%title('Dipole Radiation Pattern');

%% =========================================================
% MOC
% =========================================================

Pin = real(port.P_acc);
Zin = port.uf.tot ./ port.if.tot;
s11 = port.uf.ref ./ port.uf.inc;
SWR = (1 + abs(s11)) ./ (1 - abs(s11));

display(Zin)
display(s11)
display(SWR)
display(port.P_acc)

disp('Accepted power [W]');
disp(Pin);

%% =========================================================
% E FIELD
% =========================================================

EtDump_file = fullfile(Sim_Path, 'Et.h5');
[E_field E_mesh] = ReadHDF5Dump(EtDump_file);
Ex = E_field.FD.values{1}(:,:,:,1)*scale
Ey = E_field.FD.values{1}(:,:,:,2)*scale
Ez = E_field.FD.values{1}(:,:,:,3)*scale


% create a 2D grid to plot on
[X, Y] = ndgrid(E_mesh.lines{1},E_mesh.lines{2});
% Get E field magnitude
Eabs = sqrt(abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2);


% ------------------
Eabs(Eabs <= 0) = eps;

%% ---------------------------------------------------------
% 2D SLICE GRID
% ---------------------------------------------------------
[X, Y] = ndgrid(E_mesh.lines{1}, E_mesh.lines{2});

%% ---------------------------------------------------------
% FOCUS RANGE (REDUCE RED SATURATION)
% ---------------------------------------------------------
Emin_focus = 4;
Emax_focus = 100;

%% ---------------------------------------------------------
% LOG TRANSFORM
% ---------------------------------------------------------
Elog = log10(Eabs);

Elog_min = log10(Emin_focus);
Elog_max = log10(Emax_focus);

% clamp for better contrast in desired range
Elog(Elog < Elog_min) = Elog_min;
Elog(Elog > Elog_max) = Elog_max;

%% ---------------------------------------------------------
% PLOT
% ---------------------------------------------------------
figure;

pcolor(X, Y, Elog);
shading interp;

axis equal tight;
colormap(jet);

caxis([Elog_min, Elog_max]);

xlabel('x [m]');
ylabel('y [m]');
title('|E| field (focused 4–100 V/m range)');

%% ---------------------------------------------------------
% COLORBAR (V/m labels, log scaling underneath)
% ---------------------------------------------------------
hcb = colorbar();

ticks_phys = logspace(log10(Emin_focus), log10(Emax_focus), 10);
ticks_log  = log10(ticks_phys);

set(hcb, 'ytick', ticks_log);

labels = cell(size(ticks_phys));
for k = 1:length(ticks_phys)
    labels{k} = sprintf('%.2g', ticks_phys(k));
end

set(hcb, 'yticklabel', labels);

ylabel(hcb, '|E| [V/m]');
