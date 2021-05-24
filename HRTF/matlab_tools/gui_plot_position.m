function varargout = gui_plot_position(varargin);

% GUI_PLOT_POSITION Displays various position dependent data
%
% Usage
%   gui_plot_position(to_display_S);
%
% Input
%   to_display_S : a structure containing positions and fields to be
% displayed.
% 
% Mandatory fields of to_display_S :
%   elev_v : a column vector of elevations (degrees)
%   azim_v : a column vector of azimuths (degrees)
% User defined fields :
%   [F1], [F2],... :  Matrices or column vectors of position dependent data
% to be displayed. Every field (matrix or column vector) must have as many
% rows as position vectors (elev_v and azim_v).
%
% User Interface
%   Fields to be displayed are listed on the left pannel. Choose one field
% to be displayed.
%   Positions available are listed on the center pannel. You can either
% choose one, or several (use Shift or Ctrl keys for multipleselection).
%   Double-click on a position plots the corresponding data immediately.
%   'Select All' button allows selecting all positions.
%   You may choose between linear, 10*log10 and 20*log10 display. You may
% also choose the figure number.
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - July 2002

% Todo improve Y labels

if ishandle(varargin{1})
  % callback
  feval(varargin{2},varargin{1});
else
  % external function call
  new_gui(varargin{1});
end;

% Creates new GUI
function varargout = new_gui(varargin);

figure_h = figure('Position', [200 200 430 340], ...
      'MenuBar','none', ...
      'Resize','off', ...
      'NumberTitle','off', ...
      'Name','Plot');

% User data retrieval
params_S = varargin{1};
params_S.to_display_C = {};
field_names_C = fieldnames(varargin{1});
for field_n = 1:length(field_names_C)
  field_name_s = field_names_C{field_n};
  if ~strcmp(field_name_s,'elev_v')&~strcmp(field_name_s,'azim_v') ...
	      &isnumeric(getfield(varargin{1},field_name_s)) ...
	      &max(size(getfield(varargin{1},field_name_s))~=[1 1])
    params_S.to_display_C{end+1} = field_name_s;
  end;
end;

% Left pannel
text1_h = uicontrol('Parent',figure_h, ...
      'Style','text', ...
      'BackgroundColor',[.8 .8 .8], ...
      'String', 'Numeric Fields : ', ...
      'Position',[10 310 130 20]);
params_S.to_display_h = uicontrol('Parent',figure_h, ...
      'Style','list', ...
      'String', params_S.to_display_C, ...
      'Position',[10 10 130 300]);

% Center pannel
text2_h = uicontrol('Parent',figure_h, ...
      'Style','text', ...
      'BackgroundColor',[.8 .8 .8], ...
      'String', 'Positions [az el] : ', ...
      'Position',[150 310 130 20]);
params_S.measure_set_h = uicontrol('Parent',figure_h, ...
      'Style','list', ...
      'Max',length(params_S.elev_v), ...
      'String', num2str([params_S.elev_v params_S.azim_v]), ...
      'Position',[150 10 130 300], ...
      'Callback',['gui_plot_position(' num2str(figure_h) ',''measure_set_callback'')']);

% Right pannel
params_S.select_all_h = uicontrol('Parent',figure_h, ...
      'Style','pushbutton', ...
      'String','Select All', ...
      'Position',[290 280 130 30], ...
      'Callback',['gui_plot_position(' num2str(figure_h) ',''select_all_callback'')']);
params_S.xmode_h = uicontrol('Parent',figure_h, ...
      'Style','popupmenu', ...
      'Value',1, ...
      'String',{'Linear' '10*log10' '20*log10'}, ...
      'Position',[290 240 130 30], ...
      'Callback',['gui_plot_position(' num2str(figure_h) ',''plot_callback'')']);
text2_h = uicontrol('Parent',figure_h, ...
      'Style','text', ...
      'BackgroundColor',[.8 .8 .8], ...
      'HorizontalAlignment','left', ...
      'String', 'Figure Number : ', ...
      'Position',[290 210 100 20]);
params_S.figure_number_h = uicontrol('Parent',figure_h, ...
      'Style','edit', ...
      'HorizontalAlignment','left', ...
      'Position',[390 210 30 20]);
params_S.plot_h = uicontrol('Parent',figure_h, ...
      'Style','pushbutton', ...
      'String','Plot...', ...
      'Position',[290 170 130 30], ...
      'Callback',['gui_plot_position(' num2str(figure_h) ',''plot_callback'')']);
params_S.exit_h = uicontrol('Parent',figure_h, ...
      'Style','pushbutton', ...
      'String','Exit', ...
      'Position',[290 130 130 30], ...
      'Callback',['gui_plot_position(' num2str(figure_h) ',''exit_callback'')']);

guidata(figure_h,params_S);

% Waiting for callback
uiwait(figure_h);
% When exit (see 'exit_callback' function)
if ishandle(figure_h)
  delete(figure_h);
end;

% Select all button callback
function varargout = select_all_callback(figure_h);
      
params_S = guidata(figure_h);
set(params_S.measure_set_h,'Value',1:get(params_S.measure_set_h,'Max'));

% Double-click on a position
function varargout = measure_set_callback(figure_h);

params_S = guidata(figure_h);
if (strcmp(get(figure_h,'SelectionType'),'open'))
  plot_callback(figure_h);
end;

% Plot button callback
function varargout = plot_callback(figure_h);

params_S = guidata(figure_h);
% Selecting positions to be displayed
subset_indices_v = get(params_S.measure_set_h,'Value');
subset_elev_v = params_S.elev_v(subset_indices_v);
subset_azim_v = params_S.azim_v(subset_indices_v);
% Selecting data to be displayed
field_s = params_S.to_display_C{get(params_S.to_display_h,'Value')};
field_content_m = getfield(params_S,field_s);
field_content_m = field_content_m(subset_indices_v,:);
% X axis linear/logarithmic modes
switch get(params_S.xmode_h,'Value')
 case 2
  field_content_m = 10*log10(abs(field_content_m));
 case 3
  field_content_m = 20*log10(abs(field_content_m));
end;
% Figure number
figure_number_n = str2num(get(params_S.figure_number_h,'String'));
if isempty(figure_number_n)
  set(params_S.figure_number_h,'String',figure);
else
  figure(figure_number_n);
end;
% Position labels
labels_m = [num2str(subset_elev_v) ones(length(subset_indices_v),1)*',' ...
	    num2str(subset_azim_v)];
% Plot
if (size(field_content_m,2)==1)
  plot(field_content_m);
  set(gca,'XTick',[1:length(subset_indices_v)]);
  set(gca,'XTickLabel',labels_m);
elseif (size(field_content_m,1)==1)
  plot(field_content_m);
else  
  mesh(field_content_m);
  set(gca,'YTick',[1:length(subset_indices_v)]);
  set(gca,'YTickLabel',labels_m);
end;
guidata(figure_h,params_S);

% Exit callback function
function varargout = exit_callback(figure_h);

uiresume(figure_h);



