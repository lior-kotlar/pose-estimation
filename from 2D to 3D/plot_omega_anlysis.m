%% arrange file names

%% ####################################################################

SAVE_FIG = false ;

lists    = cell(1,2) ;
cut_dir = "G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies";
intact_dir = "G:\My Drive\Amitai\one halter experiments\roni dark 60ms";
allItems = dir(cut_dir);
lists{1} = {allItems([allItems .isdir] & startsWith({allItems .name}, 'mov')).name};

allItems = dir(intact_dir);
lists{2} = {allItems([allItems .isdir] & startsWith({allItems .name}, 'mov')).name};;

data = cell(1,2) ;
data{1} = cell(1, 1) ;
data{2} = cell(1, 1) ;

paths = [cut_dir, intact_dir];
%%
%mov_nums={'2','3','8','10','11','21','22','23','24','27','28','30','32','33','34','36','37','38','39','40','41','42','43','44','45','48','49','50','51','53','54','55','56','58','59','61','62','63','66','68','69','70','72','73','74','75'};
Ntot = sum(cellfun('length',lists)) ;
counter = 0 ;
dt = 1/16000;
for TYPE=1:2 % 1==cut, 2==intact
    N = length(lists{TYPE}) ;
    i=1;
    for mov_num=1:N
        movie_str = string(lists{TYPE}(mov_num));
        movie_h5 = strjoin([movie_str, "analysis_smoothed.h5"], "_");
        dir_path = paths(TYPE);
        h5_path = fullfile(dir_path, movie_str, movie_h5);
        if ~isfile(h5_path)
            continue
        end
        omega_body_body_frame = h5read(h5_path, "/omega_body")';
        omega_body_body_frame = omega_body_body_frame(~any(isnan(omega_body_body_frame), 2), :);
        omega_norm = sum(omega_body_body_frame.^2,2).^0.5 ;
        x_ms = (1:length(omega_norm))'*dt*1000;
        %% store data
        st.x_ms = x_ms ;
        % st.body_angs_fix = body_angs_fix ;
        st.omega_body_body_frame = omega_body_body_frame ;
        st.omega_norm = omega_norm ;
        % st.save_name = save_name ;

        data{TYPE}{i} = st ;
        i = i+1;
        clear st ;

        % %% plot euler angs
        % hfig = figure;
        % TT = tiledlayout(2,1) ;
        % TT.Padding     = "compact";
        % TT.TileSpacing = "compact" ;
        % 
        % % plot angles
        % 
        % nexttile ;
        % hold on; grid on; box on
        % yline(0,'--k','LineWidth',1,'handlevisibility','off');
        % plot(x_ms,body_angs_fix(:,[3,2,1]),'LineWidth',2)
        % set(gca,'fontsize',12)
        % %legend('yaw','pitch','roll')
        % legend('roll','pitch','yaw')
        % %xlabel('time [ms]')
        % ylabel('angle [°]')
        % 
        % disp_name = save_name ;
        % disp_name(disp_name=='_') = ' ' ;
        % 
        % title(disp_name)
        % % plot angular rates
        % nexttile ;
        % hold on; grid on; box on
        % yline(0, '--k','LineWidth',1,'handlevisibility','off');
        % plot(x_ms(2:end), omega_body_body_frame(2:end,:),'LineWidth',2)
        % hold on;
        % plot(x_ms(2:end), omega_norm(2:end),'LineWidth',4) ;
        % 
        % set(gca,'fontsize',12)
        % legend('\omega_x','\omega_y','\omega_z','|\omega|')
        % xlabel('time [ms]')
        % ylabel('angular velocity [°/sec]')
        % 
        % set(gcf,'position',[600 150 700 660]) ;
        % if SAVE_FIG
        %     print(gcf,"PNG\body_angles_" + save_name(1:end-4)  + ".png",'-dpng','-r300');
        % end
        % close(hfig)
        % 
        % counter = counter + 1 ;
        % disp(counter + " / " + Ntot)
    end
end
%close all ;

%save all_data data lists


%% distributions of |w|

fontSize = 12 ;

figure ; hold on ;
cols = [1,0,0 ; 0 1 0] ;
% edges = 0:75:10000 ;
edges = 0:100:10000 ;
cum_hist = cell(1,2) ;
cum_hist{1} = zeros(1, length(edges)-1) ;
cum_hist{2} = zeros(1, length(edges)-1) ;

N_data_points = zeros(1,2) ;

for TYPE=1:2 % 1==cut, 2==intact
    N = length(data{TYPE}) ;
    for mov_num=1:N
        st = data{TYPE}{mov_num} ;
        N_data_points(TYPE) = N_data_points(TYPE) + length(st.omega_norm(2:end)) ;
        plot(st.x_ms(1:end), st.omega_norm(1:end),'LineWidth',0.5,'color',cols(TYPE,:));
        hst = histcounts(st.omega_norm(2:end), edges) ;
        cum_hist{TYPE} = cum_hist{TYPE} + hst ;
        %plot(st.x_ms(2:end), st.omega_body_body_frame(2:end,1),'LineWidth',0.5,'color',cols(TYPE,:));
    end
end
clear hst st ;
hold off ;
box on ; grid on ;

figure ;

dw = edges(2)-edges(1) ;

hold on ;


plot(edges(1:end-1), cum_hist{2} / sum( cum_hist{2}) / dw ,'g','linewidth',2,'DisplayName','Intact') ;
plot(edges(1:end-1), cum_hist{1} / sum( cum_hist{1}) / dw ,'r','linewidth',2,'DisplayName','Cut') ;

hold off;
legend;%('Cut','Intact') ;
xlabel('Angular velocity amplitude |\omega|') ;
ylabel('Probability Density') ;
xlim([0,6000])
box on ;
set(findall(gcf,'-property','FontSize'),'FontSize',fontSize) ;
set(gcf,'color','w','position',[600 360 560 300])

%% Fourier tranform of omega_norm

figure ; hold on ;
cols = [1,0,0 ; 0 0 0] ;

omega_fourier = cell(1,2) ;
omega_fourier{1} = zeros(1, length(lists{1})) ;
omega_fourier{2} = zeros(1, length(lists{2})) ;
omega_fourier_range = omega_fourier ;

counters = [0,0] ;

Fs = 20000 ;
selected_f = 1000/15 ;
selected_f_min = 60 ;
selected_f_max = 70 ;

for TYPE=1:2 % 1==cut, 2==intact
    N = length(data{TYPE}) ;
    for mov_num=1:N
        st = data{TYPE}{mov_num} ;
        x = st.omega_norm(2:end) ;
        x = smooth(x, 2) ;
        x = x - mean(x) ;
        [ampSpec, f_half] = mySpectrum(x, Fs, false, false) ;
        ind = find(f_half>selected_f,1,'first');
        ind_range = find(f_half>=selected_f_min & f_half<=selected_f_max) ;
        omega_fourier{TYPE}(mov_num) = ampSpec(ind) ;
        omega_fourier_range{TYPE}(mov_num) = mean(ampSpec(ind_range)) ;
        plot(f_half, ampSpec ,'LineWidth',0.5,'color',cols(TYPE,:),'LineWidth',TYPE);
    end
end
clear  st ;
hold off ;
box on ; grid on ;
xlabel('frequency [Hz]');
ylabel('Amplitude spectrum')


omega_fourier_edges = 0:50:1000 ;

figure ; hold on ;
normalization = 'probability' ; % 'probability'
histogram(omega_fourier{1}, omega_fourier_edges,'FaceColor','r','normalization',normalization,'DisplayName','Cut') ;
histogram(omega_fourier{2}, omega_fourier_edges,'FaceColor','g','normalization',normalization,'DisplayName','Intact') ;
xlabel("Fourier amplitude of |\omega| at " + selected_f + "Hz (T = " + 1/selected_f*1000 + "ms)") ;
ylabel('Probability') ;
hold off ;
hold off;
% xlim([0,200])
legend;%('Cut','Intact') ;
box on ;
ytickformat('%.2f')


%% combined figure

figure('color','w') ;
TT = tiledlayout(2,1) ;
TT.TileSpacing = 'compact' ;
TT.Padding  = 'tight' ;


% plot |w|

nexttile ;
hold on ;

plot(edges(1:end-1), cum_hist{1} / sum( cum_hist{1}) / dw ,'r','linewidth',2,'DisplayName','Cut') ;
plot(edges(1:end-1), cum_hist{2} / sum( cum_hist{2}) / dw ,'g','linewidth',2,'DisplayName','Intact') ;

hold off;
legend;
xlabel('Angular velocity amplitude |\omega|') ;
ylabel('Probability Density') ;
xlim([0,6000])
box on ;

% plot Fourier amplitude

nexttile
hold on ;
normalization = 'probability' ; % 'probability'
histogram(omega_fourier{1}, omega_fourier_edges,'FaceColor','r','normalization',normalization, ...
    'DisplayName','Cut') ;
histogram(omega_fourier{2}, omega_fourier_edges,'FaceColor','g','normalization',normalization,'DisplayName','Intact') ;
xlabel("Fourier amplitude of |\omega| at " + round(selected_f) + "Hz (T = " + 1/selected_f*1000 + "ms)") ;
ylabel('Probability') ;
hold off ;
hold off;
xlim([0,900])
legend;%('Cut','Intact') ;
box on ;
ytickformat('%.1f')

% format all
set(gcf,'color','w','position',[600 160 560 600])
set(gcf,'units','centimeters') ;
set(gcf,'position',[15 4 8 9.6 ])

set(findall(gcf,'-property','FontSize'),'FontSize',9) ;

if SAVE_FIG
    print(gcf,'mosquito_haltere_cut.png','-dpng','-r300') ;
    exportgraphics(gcf,'mosquito_haltere_cut.pdf')
end
return


    %{
%% standard deviation of |w| per movie

omega_std = cell(1,2) ;
omega_std{1} = zeros(1, length(lists{1})) ;
omega_std{2} = zeros(1, length(lists{2})) ;

for TYPE=1:2 % 1==cut, 2==intact
    N = length(lists{TYPE}) ;
    for mov_num=1:N  
        st = data{TYPE}{mov_num} ;
        if length(st.x_ms) < 80*20
            fac = NaN ;
        else 
            fac = 1 ;
        end
        omega_std{TYPE}(mov_num) = fac*std(st.omega_norm(2:end)) ;        
    end
end
omega_std_edges = 0:40:1000 ;
figure ; hold on ;
hst = histcounts(st.omega_norm(2:end), edges) ;
%histogram(omega_std{1},omega_std_edges,'FaceColor','r') ;
%histogram(omega_std{2},omega_std_edges,'FaceColor','g') ;
xlabel('Standard deviation of |\omega|') ;
ylabel('Counts') ;
hold off ; 
hold off;
legend('Cut','Intact') ; 
box on ;

%% autocorrelation of body axis unit vector

for TYPE=1:2 % 1==cut, 2==intact
    N = length(lists{TYPE}) ;
    for mov_num=1:N  
        st = data{TYPE}{mov_num} ;
        st.body_angs_fix;
    end
end

    %}