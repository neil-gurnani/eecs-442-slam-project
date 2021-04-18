%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ x_k_k, p_k_k, featuresInfo ] = intializeNewFeature( cam, im, step,...
    std_pxl, lambdaInit, std_lambda, x_k_k, p_k_k, figure_all, im_fig, featuresInfo )

halfFeatPatchSize = 10;

% Feature selection by clicking
while(1),
    figure(figure_all);
    subplot(im_fig);
    hold on;
    [click_u, click_v, button] = ginput(1);    
    if (button==1)&&(click_u>halfFeatPatchSize+1)&&(click_u<cam.nCols-halfFeatPatchSize-1)...
            &&(click_v>halfFeatPatchSize+1)&&(click_v<cam.nRows-halfFeatPatchSize-1)
        break;
    end
end

plot( click_u, click_v, 'g*' );

% Add the feature to the vector state
click_u = click_u-0.5;
click_v = click_v-0.5;
[ x_k_k, p_k_k ] = addNewFeatures_newPar( [click_u;click_v], x_k_k, p_k_k, cam, std_pxl, lambdaInit, std_lambda );

% Save patch information in order to recognize the feature
index = size(featuresInfo,2)+1;
featuresInfo(index).patch_p_f =...
    double(im(round(click_v+0.5)-halfFeatPatchSize:round(click_v+0.5)+halfFeatPatchSize,...
    round(click_u+0.5)-halfFeatPatchSize:round(click_u+0.5)+halfFeatPatchSize));
featuresInfo(index).positionInStateVector = (size(x_k_k,1)-13)/6;