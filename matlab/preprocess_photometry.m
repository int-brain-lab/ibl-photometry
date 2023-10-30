clear all
close all

% This script is an example preprocessing pipeline for photometry data
% recorded using the Doric photometry system. All raw files contain the
% data of two animals tested in parallel (same stimulation intensity of
% about 50uW at fiber tips) in an open field with or without an object in
% it (1412 without, 1512 with). 
% it implements:
% - a correction for temporal drifts using a polynomial fit of
% order 2
% - a correction for nuisance variables (based on the isosbestic
% channel) using a robustfit approach and including a quadratric term
% (which tends to improve things)
% - a calculation of a pseudo dF/F0 which makes use of the constant term of
% the polynomial fit to recover the "baseline" signal after correction
% steps
% - calculation of signal skewness after drift correction and after
% nuisance correction. Rightward (i.e positive) skewness in photometry data
% usually indicates the presence of transients. Sessions with a final
% skewness superior to 0.8 or 1 are almost always "good enough" sessions.
% Note that the script also tries to find the optimal backward lag for the 
% moving mean filter by optimizing the resulting skewness value. It is
% experimental and probably not worth using but it somehow validate the
% order of magnitude used by default for this lag
% In terms of outcome, it is quite clear that animals A, B, F and H are
% good animals. D and E have no or very little signal. I and J have some
% signal on different days, which is a bit odd.
% Information about the mice
% - A: ai95SERT-GcAMP6f, DRN -4.6/-3.05/0.05, lateral angle 15 (extra +0.1 DV)
% - B: ai148SERT-GcAMP6f, DRN -4.6/-3.1/0, backward angle (extra 0.2 DV)
% - D: ai148SERT-GcAMP6f, DRN -4.6/-3.05/0.05, lateral angle 10 (extra -0.1/-0.15 DV)
% - E: ai148SERT-GcAMP6f, DRN -4.6/-3.05/0.05, lateral angle 20 (extra -0.1/-0.15 DV)
% - F: ai148SERT-GcAMP6f, DRN -4.6/-3.1/0.05, backward angle 30 (extra +0.1 DV)
% - H: ai148DAT-GcAMP6f, VTA -3.1/-4/0.5 (need to double check)
% - I: ai148DAT-GcAMP6f, TS -1.5/-2.5/3.25 (extra +0.1)
% - J: ai148DAT-GcAMP6f, TS -1.5/-2.5/3.25 (extra +0.1)
% For D and E, I stopped the insertion before target because I started to
% see a clear increase in photometry signal (as in A) and I thought that it
% might be enough to stop. It was likely a mistake.
% For I and J, the large artifacts likely come from the fact that a
% relatively long segment of the fiber coating is exposed to light, or to
% the fact that the animals manage to touch the fiber when scratching their
% head.

% specifications
resample_fs=80; % in Hz
polyorder=2;
filtparam=[-0.05 0]; % moving average lag in seconds
backward_optimarray=-0.5:0.005:0; 
% input files
rawfolder='raw_photodata';
filelist=dir([rawfolder '/*.csv']);

% output files
preprocfolder='preproc_photodata';
mkdir(preprocfolder);
figfolder='preproc_photodata_figures';
mkdir(figfolder)

for f=1:length(filelist)
    
    % this part is specific to the file naming/structure and should be
    % adapted to your file configuration
    % 
    fname=[rawfolder '/' filelist(f).name];
    parsename=strfind(filelist(f).name,'_');
    fdate=filelist(f).name(1:parsename(1)-1);
    fmice={filelist(f).name(parsename(2)+1:parsename(3)-1), filelist(f).name(parsename(3)+1:parsename(4)-1)};
    fdata=importdata(fname);
    
    % remove completely irrelevant columns
    fdata.data(:,10:13)=[];
    
    % relevant columns (based on fdata.textdata or .colheader information)
    seccol=1; % time
    g475col=[2 5]; % gcamp 465
    g405col=[3 6]; % gcamp 405
    ttlcol=[8 9]; % ttl for sync with video
    
    % compute actual fs
    sec=fdata.data(:,seccol);
    fs=1/((sec(end)-sec(1))/length(sec)); 
    
    % resample all the data to desired frequency using 
    % linear interpolation
    rs_time=sec(1):1/resample_fs:sec(end);
    rs_data=interp1(sec,fdata.data,rs_time,'linear');
    
    % fix interpolated TTL to ensure binary values.
    for c=1:length(ttlcol)
        rs_data(:,ttlcol(c))=double(rs_data(:,ttlcol(c))>0.5);
    end
    
    for ff=1:2
        
        % allocate
        secs=rs_data(:,seccol);
        g475=rs_data(:,g475col(ff));
        g405=rs_data(:,g405col(ff));
        ttl=rs_data(:,ttlcol(ff));
        
        raw=table(secs,g475,g405,ttl);

        % simply crop nans at the beginning and at the end of the recording
        % make sure that we don't crop more than one second on each
        % extremity and that we don't crop a ttl
        nanind=find(isnan(g475) | isnan(g405));       
        assert(sum(nanind<resample_fs & nanind>length(rs_data-resample_fs))==0,'more nans than expected: double-check')
        assert(sum(ttl(nanind))==0,'cropping ttl is not good');
        raw(nanind,:)=[];
        
        % crop everything before first ttl and everything after the last
        % one
        ttlsamples=find(diff(raw.ttl)==1)+1;
        ttlstart=ttlsamples(2);
        ttlend=ttlsamples(end);
        
        raw=raw(ttlstart:ttlend,:);
        
        % in what follows, we treat the g475/405 channels keeping the name
        % identical between different processing stage (i.e. g475cor for
        % 'corrected'). This allows to simply skip/comment a step without
        % disrupting the process.
        g475cor=raw.g475;
        g475cor=raw.g405;        
        offset=mean(raw.g475);
        
        %% time correction using polynomial fitting

        figure('name', ['drift correction -' fmice{ff} '-' fdate],'color','w', 'units','normalized','position', [0.2 0.2 0.6 0.6])
        % 475 channel
        [p475,~,mu]=polyfit(raw.secs,raw.g475,polyorder);
        offset=p475(end);
        f475=polyval(p475,raw.secs,[],mu);
        g475cor=raw.g475-f475;
        subplot(2,1,1)
        plot(zscore([g475cor,f475,raw.g475])); % zscoring for visual convenience 
        title('poly475');
        legend({'cor','fit','raw'});
        xlabel('time (samples)')
        ylabel('zscored signal')
        
        % 405 channel
        [p405,~,mu]=polyfit(raw.secs,raw.g405,polyorder);
        f405=polyval(p405,raw.secs,[],mu);
        g405cor=raw.g405-f405;
        subplot(2,1,2)
        plot(zscore([g405cor,f405,raw.g405])); % zscoring for visual convenience 
        title('poly405');
        legend('cor','fit','raw');
        xlabel('time (samples)')
        ylabel('zscored signal')
        
        saveas(gcf,[figfolder '/' get(gcf,'name') '.png']);
        
        % plot the distributions of the signal: good recording should show
        % a distribution more skewed to the right (i.e. positive) for the 475
        % (blue) channel than for the 405 (violet) channel.
        % These are probably our transients!
        % The distributions should look more or less Gaussian if there are no jumping artifacts.
        figure('name', ['initial signal distribution 1' fmice{ff} '-' fdate],'color','w', 'units','normalized','position', [0.2 0.2 0.6 0.6])
        histogram(zscore(g405cor),50, 'facealpha',0.5, 'facecolor',[0.5 0 1]);
        hold on
        histogram(zscore(g475cor),50, 'facealpha',0.5, 'facecolor',[0 0 1]);
        annotation('textbox',[0.6 0.6 0.3 0.3],'String',{['skewness 405: ' num2str(skewness(zscore(g405cor)))];['skewness 475: ' num2str(skewness(zscore(g475cor)))]},'FitBoxToText','on');
        box off
        xlabel('zscored signal')
        ylabel('count')
        
        saveas(gcf,[figfolder '/' get(gcf,'name') '.png'])


        %% regression of nuisance signals
        Xnuis=[g405cor g405cor.*g405cor g405cor.^3];
        y=g475cor;
        bnuisance=robustfit(Xnuis,y); % glmfit() also possible
        fnuis=glmval(bnuisance,Xnuis,'identity');
        g475cor=g475cor-fnuis;
        
        figure('name', ['nuisance correction - ' fmice{ff} '-' fdate],'color','w', 'units','normalized','position', [0.2 0.2 0.6 0.6])
        plot(zscore([g475cor,fnuis])); % zscoring for visual convenience only
        legend('cor','fit');   
        xlabel('time (samples)')
        ylabel('zscored signal')
        
        saveas(gcf,[figfolder '/' get(gcf,'name') '.png'])


        
        %% skewness maximization (experimental/custom)
        % if positive skewness is a good proxy for the signal of interest,
        % we may use it to optimize of filtering without inducing artifacts
        skewfunc= @(p) skewness(movmean(zscore(g475cor),round(resample_fs*abs(p))));
        search_backlag=[backward_optimarray; 0*backward_optimarray];
        search_skew=nan(1,size(search_backlag,2));
        for i=1:length(search_backlag);
            search_skew(i)=skewfunc(search_backlag(:,i)');
        end
        minind=find(search_skew==max(search_skew));
        best_backlag(f,ff)=median(search_backlag(1,minind));
%         figure % if the skewness(lag) function has more than one peak, be extra-mindful
%         plot(search_backlag(1,:),search_skew);
        g475corsk=movmean(g475cor, abs(round(resample_fs*[best_backlag(f,ff) 0])));
        
        
        %% moving average filtering
        g475cor=movmean(g475cor, round(resample_fs*abs(filtparam)));
        
        %% pseudo deltaF/F0 (as in Seo and Warden);
        dFovF0=100*(g475cor)/mean((g475cor+offset));
        
        %% final trace
        figure('name', ['preprocessed -' fmice{ff} '-' fdate],'color','w', 'units','normalized','position', [0.2 0.2 0.6 0.6])
        subplot(3,1,1)
        plot(raw.secs,1000*g475cor,'linewidth',.5,'color', 'k')
        try
            hold on
            plot(raw.secs,g475corsk*1000, 'linewidth',1, 'color', [0.1 0.9 0.1])
        end
        xlabel('time (s)')
        ylabel('mV')
        subplot(3,1,2)
        plot(raw.secs,dFovF0,'linewidth',.5,'color', 'k')
        xlabel('time (s)')
        ylabel('dF/F0 (%)')
        subplot(3,1,3)
        histogram(zscore(g475cor),50, 'facealpha',0.5, 'facecolor',[0 0 1]);
        annotation('textbox',[0.6 0.0 0.3 0.3],'String',{['final skewness 475: ' num2str(skewness(zscore(g475cor)))];[ 'final skewness opti475: ' num2str(skewness(zscore(g475corsk)))]},'FitBoxToText','on');
        box off
        title('final signal distribution')
        xlabel('zscored signal')
        xlabel('count')
        saveas(gcf,[figfolder '/' get(gcf,'name') '.png'])

    end
    
end
        
        % 

        
        