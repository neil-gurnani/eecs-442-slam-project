%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ X_RES, P_RES ] = addNewFeatures_newPar( uvd, X, P, cam, std_pxl, lambdaInit, std_lambda )
% Add a new feature to the vector state

% Camera state
Xv = X(1:13);

newFeature = hinv( uvd, Xv, cam, lambdaInit );
X_RES = [ X; newFeature ];
P_RES = addAFeatureCov_newPar( P, uvd, Xv, lambdaInit, std_pxl, std_lambda, cam );
