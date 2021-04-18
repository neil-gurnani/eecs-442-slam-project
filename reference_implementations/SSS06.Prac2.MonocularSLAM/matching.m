%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [z, h, H_matched, R_matched, matched_all ] = matching( predicted, S_predicted, ...
    H_predicted, std_pxl, featuresInfo, cam, im, chi2inv_table )

halfPatchSize = 10;
matchThreshold = 0.8;

predicted_size = size( predicted, 1 );
predicted_image = floor( predicted ) + ones( predicted_size, size( predicted, 2 ) );

matched_all = ones( predicted_size, 2 )*(-1);

% for every predicted feature
for li=1:predicted_size
    
    if predicted(li,1)>0
        
        S_i = S_predicted(1:2,1:2);
        S_predicted = S_predicted(3:end,3:end);
        inv_S_i = inv(S_i);
        halfW_x = ceil(2*sqrt(S_i(1,1)));
        halfW_y = ceil(2*sqrt(S_i(2,2)));
        correlationScore = zeros((2*halfW_x+1)*(2*halfW_y+1),3);
        i_corrScore = 1;
        % for the acceptance region
        for j_k = predicted_image(li,1)-halfW_x : predicted_image(li,1)+halfW_x
            for i_k = predicted_image(li,2)-halfW_y : predicted_image(li,2)+halfW_y
                if((j_k>halfPatchSize+1) && (j_k<(cam.nCols-halfPatchSize-1)) &&...
                        (i_k>halfPatchSize+1) && (i_k<(cam.nRows-halfPatchSize-1)))
                    innov = [ j_k-predicted(li, 1);i_k-predicted(li, 2) ];
                    if((innov'*inv_S_i*innov)<chi2inv_table(1, 3))
                        % compute correlation
                        newImagePatch = im( i_k-halfPatchSize:i_k+halfPatchSize, j_k-halfPatchSize:j_k+halfPatchSize );
                        correlationScore(i_corrScore,:)=[j_k,i_k,crosscorr(newImagePatch,featuresInfo(li).patch_p_f)];
                        i_corrScore = i_corrScore+1;
                    end
                end
            end
        end
        % select maximum correlation value
        [ maxCorr, maxIndex ] = max( correlationScore( :,3 ) );
        if( maxCorr > matchThreshold )
            matched_all( li, : )=correlationScore( maxIndex, 1:2 );
        end
        
    end
    
end

% Construct results

matched_index = matched_all(:,1) ~= -1;
    
z = matched_all( matched_index, : ) ;
z = reshape( z',size(z,1)*size(z,2), 1 );
z = z-0.5*ones( size(z,1), 1 );

z_size = size( z, 1 );

h = predicted(matched_index,:) ;
h = reshape(h',size(h,1)*size(h,2),1);

predicted_index = predicted(:,1) ~=-1;
H_matched_index = matched_index( predicted_index, : );
H_matched_index = reshape( [H_matched_index H_matched_index]', 2*size(H_matched_index,1), 1 );
H_matched = H_predicted( H_matched_index, : );

R_matched = eye( z_size )*std_pxl^2;