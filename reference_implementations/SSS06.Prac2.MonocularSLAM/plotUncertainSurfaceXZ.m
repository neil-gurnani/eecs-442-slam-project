%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotUncertainSurfaceXZ( C, nu, chi2, color, randSphere6D, nPointsRand )

% jcivera, 12/1/06

K = chol( C )';

XYZTPL = K*randSphere6D +...
    [ ones(1,nPointsRand)*nu(1); ones(1,nPointsRand)*nu(2); ones(1,nPointsRand)*nu(3);...
        ones(1,nPointsRand)*nu(4); ones(1,nPointsRand)*nu(5); ones(1,nPointsRand)*nu(6)];

lambdaPositiveIndex = find(XYZTPL(6,:)>0);
XYZTPL_LPositive = XYZTPL(:,lambdaPositiveIndex);

if ~isempty(XYZTPL_LPositive) && size(lambdaPositiveIndex,2)>10
    
    for j = 1:size(XYZTPL_LPositive,2)
        XYZ(:,j) = XYZTPL2XYZ( XYZTPL_LPositive(:,j) );
    end
    
    k = convhull( XYZ(1,:)', XYZ(3,:)' );
    plot3( XYZ(1, [k ; k(1)]), zeros(1,size(k,1)+1), XYZ(3, [k ; k(1)]), 'color',color, 'LineWidth', 1.5);
    hold on;
    
end