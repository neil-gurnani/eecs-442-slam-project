%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the parameters of a camera
% Javier Civera, 8/11

function cam = setCamParameters()

d =     0.0112;
nRows = 240;
nCols = 320;
Cx =    1.7012/d;
Cy =    1.5138/d;
k1=     6.931e-2;
k2=     1.092e-2;
f =     2.2014;

cam.k1 =    k1;
cam.k2 =    k2;
cam.nRows = nRows;
cam.nCols = nCols;
cam.Cx =    Cx;
cam.Cy =    Cy;
cam.f =     f;
cam.dx =    d;
cam.dy =    d;
cam.model = 1;


cam.K =     [ -f/d   0    Cx;
                0  -f/d   Cy;
                0    0     1];