%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Numerical values and graphics initialisation

setpath;
warning('off');

% Sequence path and initial image
handles.sequencePath = 'juslibol/seqtest';
handles.initialImage = 2;
handles.numberOfImages = 180;

% General
handles.step = 1;

handles.lambdaInit = 0.1;
handles.std_lambda = 0.5;

% Initial state vector and covariance matrix
handles.x_k_k = [0 0 0 1 0 0 0 eps eps eps eps eps eps]';
handles.p_k_k = diag( [ eps eps eps eps eps eps eps 1 1 1 0.5 0.5 0.5 ] );

% Camera parameters
handles.cam = setCamParameters;

% Figures
handles.figure_all = figure;
oldUnits = get( handles.figure_all, 'Units' );
set( handles.figure_all, 'Units', 'pixels' );
set( handles.figure_all, 'Position', [50,50,768,384] );
set( handles.figure_all, 'Units', oldUnits );

handles.figure_image = subplot('position',[0.04 0.1875 0.4 0.625]);
title('Pred&Matched:Red. Pred&NOTMatched:Cian. Match:Green');
handles.figure_3D = subplot('position',[0.5 0.1875 0.4 0.625]);
xlabel('X[m]');
ylabel('Y[m]');
zlabel('Z[m]');
title('Top view of the 3D world. Predicted:red. Camera:Blue');
view(-180,0);

% chi squared table
handles.chi2inv_table = [0.95 2 5.99146454710798;
                         0.95 3 7.81472790325116;
                         0.95 6 12.59158724374398];

% random points to draw the points covariance
nPointsRand = 1000;
X = rand( 1, nPointsRand )-0.5;
Y = rand( 1, nPointsRand )-0.5;
Z = rand( 1, nPointsRand )-0.5;
theta = rand( 1, nPointsRand )-0.5;
phi = rand( 1, nPointsRand )-0.5;
lambda = rand( 1, nPointsRand )-0.5;
for i = 1:nPointsRand
    a = [X(i) Y(i) Z(i) theta(i) phi(i) lambda(i)];
    a = a/norm(a)*sqrt(handles.chi2inv_table(3,3));
    X(i) = a(1); Y(i) = a(2); Z(i) = a(3);
    theta(i) = a(4); phi(i) = a(5); lambda(i) = a(6);
end
handles.randSphere6D = [ X; Y; Z; theta; phi; lambda ];
handles.nPointsRand = nPointsRand;

% gui variables
handles.camHistory = zeros(6,900);
handles.featuresInfo = [];
handles.Xmin = -15;
handles.Xmax = 15;
handles.Zmin = -15;
handles.Zmax = 15;