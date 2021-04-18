%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ X_km1_k, P_km1_k ] = prediction( X_k, P_k, delta_t, SD_A_component_filter, SD_alpha_component_filter )
% Prediction step

% Camera motion prediction
Xv_km1_k = fv( X_k(1:13,:), delta_t );

% The prediction for the features is to stay at the same place
X_km1_k = [ Xv_km1_k; X_k( 14:end,:) ];

% Prediction for P
F = dfv_by_dxv( X_k(1:13,:),zeros(6,1),delta_t );

Pn=diag([(SD_A_component_filter*delta_t)^2 (SD_A_component_filter*delta_t)^2 (SD_A_component_filter*delta_t)^2 ...
        (SD_alpha_component_filter*delta_t)^2 (SD_alpha_component_filter*delta_t)^2 (SD_alpha_component_filter*delta_t)^2]);

[Q,G] = func_Q( X_k(1:13,:), zeros(6,1), Pn, delta_t );

nFeat=(size(X_k,1)-13)/3;

F_tot     = [F                 zeros(13,3*nFeat);
            zeros(13,3*nFeat)'      eye(3*nFeat)];
    
P_km1_k = F_tot*P_k*F_tot'      +    [ Q                       zeros(13,3*nFeat);
                                       zeros(13,3*nFeat)'       zeros(3*nFeat,3*nFeat)];