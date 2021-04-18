%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function im = getImage( image_file_name_prefix, k )
% Reads an image from the disk

imRGB = imread( sprintf( '%s%04d.pgm', image_file_name_prefix, k ) );
im = double(imRGB( :, :, 1 ));