%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLAM Summer School 2006, Oxford.
% Practical 3. SLAM using Monocular Vision.
% Practical exercise.
% J.M.M. Montiel, Javier Civera, Andrew J. Davison.
% {josemari, jcivera}@unizar.es, ajd@doc.ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function varargout = mono_slam(varargin)
% MONO_SLAM M-file for mono_slam.fig
%      MONO_SLAM, by itself, creates a new MONO_SLAM or raises the existing
%      singleton*.
%
%      H = MONO_SLAM returns the handle to a new MONO_SLAM or the handle to
%      the existing singleton*.
%
%      MONO_SLAM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MONO_SLAM.M with the given input arguments.
%
%      MONO_SLAM('Property','Value',...) creates a new MONO_SLAM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before mono_slam_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to mono_slam_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help mono_slam

% Last Modified by GUIDE v2.5 24-Aug-2006 20:49:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @mono_slam_OpeningFcn, ...
                   'gui_OutputFcn',  @mono_slam_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin & isstr(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before mono_slam is made visible.
function mono_slam_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to mono_slam (see VARARGIN)

% Choose default command line output for mono_slam
handles.output = hObject;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the parameters of the simulation,
% storing them in the data structure handles.
setParameters;

% Parameters you have to modify during the practical
% rho history data
handles.rhoHistory_5 = [];
handles.rhoHistory_15 = [];
handles.feature5History = [];
handles.rhoStdHistory_5 = [];
handles.rhoStdHistory_15 = [];
% Covariance definitions
handles.sigma_imageNoise = 1; % pixels
handles.sigma_aNoise     = 6; % m s^-2
handles.sigma_alphaNoise = 6; % rad s^-2
% Time between frames
handles.deltat = 1/30; %s

% Show the image, to select features in the first step
im = getImage( handles.sequencePath, handles.initialImage );
figure( handles.figure_all );
subplot(handles.figure_image);
hold off;
imagesc(im);
colormap gray;
hold on;
title('Pred&Matched:Red. Pred&NOTMatched:Cian. Match:Green');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(handles.state_text, 'String', 'Ready...');


% Update handles structure
guidata(hObject, handles);

% UIWAIT makes mono_slam wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = mono_slam_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in EKF_step.
function EKF_step_Callback(hObject, eventdata, handles)
% hObject    handle to EKF_step (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.state_text, 'String', 'Running, please wait...');

% Extract needed data from the data structure handles
sequencePath = handles.sequencePath;
initialImage = handles.initialImage;
step = handles.step;
x_k_k = handles.x_k_k;
p_k_k = handles.p_k_k;
deltat = handles.deltat;
sigma_aNoise = handles.sigma_aNoise;
sigma_alphaNoise = handles.sigma_alphaNoise;
sigma_imageNoise = handles.sigma_imageNoise;
cam = handles.cam;
featuresInfo = handles.featuresInfo;
camHistory = handles.camHistory;
figure_all = handles.figure_all;
figure_image = handles.figure_image;
figure_3D = handles.figure_3D;
randSphere6D = handles.randSphere6D;
nPointsRand = handles.nPointsRand;
chi2inv_table = handles.chi2inv_table;
rhoHistory_5 = handles.rhoHistory_5;
rhoHistory_15 = handles.rhoHistory_15;
rhoStdHistory_5 = handles.rhoStdHistory_5;
rhoStdHistory_15 = handles.rhoStdHistory_15;
feature5History = handles.feature5History;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Monocular EKF step

if (initialImage + step-1) <= handles.numberOfImages
    
    % get an image
    im = getImage( sequencePath, initialImage + step-1 );
    
    % prediction
    [ x_km1_k, p_km1_k ] = prediction( x_k_k, p_k_k, deltat, sigma_aNoise, sigma_alphaNoise );
    
    % measurements prediction
    [ h_all, H_predicted, S ] = measurements_prediction( x_km1_k, p_km1_k, sigma_imageNoise, cam );
    
    % matching
    [ z, h, H_matching, R_matching, z_all ] = matching( h_all, S,...
        H_predicted, sigma_imageNoise, featuresInfo, cam, im, chi2inv_table );
    
    % update
    [ x_k_k, p_k_k ] = update( x_km1_k, p_km1_k, H_matching, R_matching, z, h );
    
    % save camera location
    camHistory(:,step) = [x_k_k(1:3); tr2rpy(q2tr(x_k_k(4:7)))'];
    
    % Here, the code to save history information of two features.
    if size( x_k_k, 1 ) >= (13+5*6)
        rho_estimated = x_k_k( (13+5*6) );
        feature5 = x_k_k( (13+5*6-5:13+5*6) );
        sigma_estimated = sqrt(p_k_k( (13+5*6), (13+5*6) ));
        rhoHistory_5 = [ rhoHistory_5, rho_estimated ];
        rhoStdHistory_5 = [ rhoStdHistory_5, sigma_estimated ];
        feature5History = [feature5History, feature5];
        set(handles.feature5estimation, 'String', sprintf( '[%6.3g, %6.3g, %6.3g]',...
            1/(rho_estimated + 2*sigma_estimated),...
            1/rho_estimated,...
            1/(rho_estimated - 2*sigma_estimated) ));
    end
    if size( x_k_k, 1 ) >= (13+15*6)
        rho_estimated = x_k_k( (13+15*6) );
        sigma_estimated = sqrt(p_k_k( (13+15*6), (13+15*6) ));
        rhoHistory_15 = [ rhoHistory_15, rho_estimated ];
        rhoStdHistory_15 = [ rhoStdHistory_15, sigma_estimated ];
        set(handles.feature15estimation, 'String', sprintf( '[%6.3g, %6.3g, %6.3g]',...
            1/(rho_estimated + 2*sigma_estimated),...
            1/rho_estimated,...
            1/(rho_estimated - 2*sigma_estimated) ));
    end
    save history.mat rhoHistory_5 rhoStdHistory_5 rhoHistory_15 rhoStdHistory_15 feature5History
    
    % plots
    plotresults( step, figure_all, figure_image, figure_3D,...
        x_k_k, p_k_k, h_all, S, z_all, featuresInfo, cam, im,...
        camHistory(:,1:step), randSphere6D, nPointsRand,...
        [handles.Xmin handles.Xmax -5 5 handles.Zmin handles.Zmax], chi2inv_table );
    
    step = step+1;
    
else
    
    plotresults( step, figure_all, figure_image, figure_3D,...
        x_k_k, p_k_k, [], [], [], featuresInfo, cam, [],...
        camHistory(:,1:step), randSphere6D, nPointsRand,...
        [handles.Xmin handles.Xmax -5 5 handles.Zmin handles.Zmax], chi2inv_table );
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% store data for the next step
handles.step = step;
handles.x_k_k = x_k_k;
handles.p_k_k = p_k_k;
handles.featuresInfo = featuresInfo;
handles.camHistory = camHistory;
handles.rhoHistory_5 = rhoHistory_5;
handles.rhoHistory_15 = rhoHistory_15;
handles.rhoStdHistory_5 = rhoStdHistory_5;
handles.rhoStdHistory_15 = rhoStdHistory_15;
handles.feature5History = feature5History;

set(handles.state_text, 'String', sprintf('Ready... %d/%d', step, handles.numberOfImages - initialImage - 1));

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in Add_feature.
function Add_feature_Callback(hObject, eventdata, handles)
% hObject    handle to Add_feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Extract needed data from the data structure handles
step = handles.step;
sequencePath = handles.sequencePath;
initialImage = handles.initialImage;
x_k_k = handles.x_k_k;
p_k_k = handles.p_k_k;
sigma_imageNoise = handles.sigma_imageNoise;
cam = handles.cam;
featuresInfo = handles.featuresInfo;
figure_all = handles.figure_all;
figure_image = handles.figure_image;
lambdaInit = handles.lambdaInit;
std_lambda = handles.std_lambda;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializes a new feature
[ x_k_k, p_k_k, featuresInfo ] = intializeNewFeature( cam, getImage( sequencePath, initialImage + step-2 ), step,...
            sigma_imageNoise, lambdaInit, std_lambda, x_k_k, p_k_k, figure_all, figure_image, featuresInfo );
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MONO_SLAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
% store data for the next step
handles.x_k_k = x_k_k;
handles.p_k_k = p_k_k;
handles.featuresInfo = featuresInfo;

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Xmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Xmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function Xmin_Callback(hObject, eventdata, handles)
% hObject    handle to Xmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Xmin as text
%        str2double(get(hObject,'String')) returns contents of Xmin as a double

handles.Xmin = str2double(get(hObject,'String'));

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Zmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Zmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function Zmax_Callback(hObject, eventdata, handles)
% hObject    handle to Zmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Zmax as text
%        str2double(get(hObject,'String')) returns contents of Zmax as a double

handles.Zmax = str2double(get(hObject,'String'));

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Zmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Zmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function Zmin_Callback(hObject, eventdata, handles)
% hObject    handle to Zmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Zmin as text
%        str2double(get(hObject,'String')) returns contents of Zmin as a double

handles.Zmin = str2double(get(hObject,'String'));

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function Xmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Xmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function Xmax_Callback(hObject, eventdata, handles)
% hObject    handle to Xmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Xmax as text
%        str2double(get(hObject,'String')) returns contents of Xmax as a double

handles.Xmax = str2double(get(hObject,'String'));

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in Exit.
function Exit_Callback(hObject, eventdata, handles)
% hObject    handle to Exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

warning('on');
close all;
clear variables;
clc;