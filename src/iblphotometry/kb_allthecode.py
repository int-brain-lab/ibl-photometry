#%%
"""
KceniaBougrova 
08October2024 

1. LOAD THE BEHAVIOR AND PHOTOMETRY FILES
2. ADD BEHAVIOR VARIABLES 
3. SYNCHRONIZE BEHAV AND PHOTOMETRY 
4. PREPROCESS PHOTOMETRY
5. PLOT HEATMAP AND LINEPLOT DIVIDED BY FEEDBACK TYPE 

""" 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from brainbox.behavior.training import compute_performance 
from brainbox.io.one import SessionLoader 
import iblphotometry.kcenia as kcenia 
import ibldsp.utils
import scipy.signal 
import re
from iblutil.numerical import rcoeff
from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
from one.api import ONE #always after the imports 
one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir="/mnt/h0/kb/data/one", mode='remote')

""" useful""" 
# eids = one.search(project='ibl_fibrephotometry') 
#%% ####################################################################################################################
""" EDIT THE VARS - eid, ROI, photometry file path (.csv or .pqt) """ 
eid = '47c02dcc-337b-4964-937f-39928c057fff' #example eid 
# eid = 'a1ccc8ed-9829-4af8-91fd-cc1c83b74b98' 
""" LOAD TRIALS """ 
df_trials, subject, session_date = kcenia.load_trials_updated(eid, one=one) 

region_number = "3" #the ROI number you recorded from 
nph_bnc = 0 #or 1, the BNC input you use to sync the data; "Input0" or "Input1" 
#choose one: .csv or .pqt 
base_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{subject}/{session_date}/'
available_dirs = os.listdir(base_path) # List all directories in the base path
match = [d for d in available_dirs if re.match(r'^00\d$', d)] # Use regex to match the directories that start with 00 and end with a digit

if match:
    matched_dir = match[0]     # Select the first match (assuming only one 00X folder exists)
    nph_file_path = os.path.join(base_path, matched_dir, 'raw_photometry_data', 'raw_photometry.csv')     # Construct the full file path with the matched directory
    df_nph = pd.read_csv(nph_file_path)
    # df_nph = pd.read_parquet(nph_file_path)
else:
    print(f"No directories matching the pattern '00X' were found in {base_path}")

""" SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
EVENT = "feedback_times" 
time_bef = -1
time_aft = 2

# ##### if you dont know the eid, do it by mouse and date #####
# mouse = "ZFM-06275"
# date = '2023-09-18'
#eid = get_eid(mouse, date) 

#%% #########################################################################################################
""" GET PHOTOMETRY DATA """ 
region = f"Region{region_number}G"
df_nph["mouse"] = subject
df_nph["date"] = session_date
df_nph["region"] = region
df_nph["eid"] = eid 

"""
CHANGE INPUT AUTOMATICALLY 
""" 
tbpod = df_trials['stimOnTrigger_times'].values #bpod TTL times
iup = ibldsp.utils.rises(df_nph[f'Input{nph_bnc}'].values) #idx nph TTL times 
tph = (df_nph['Timestamp'].values[iup] + df_nph['Timestamp'].values[iup - 1]) / 2 #nph TTL times computed for the midvalue 
fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
if len(tph)/len(tbpod) < .9: 
    print("mismatch in sync, will try to add ITI duration to the sync")
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].values,
        df_trials['intervals_1'].values - 1,  # here is the trick
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
    if len(tph)/len(tbpod) > .9:
        print("still mismatch, maybe this is an old session")
        tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing #2")
assert abs(drift_ppm) < 100, "drift is more than 100 ppm"

df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"]) 

fcn_nph_to_bpod_times(df_nph["Timestamp"])

df_nph["Timestamp"]

df_nph = kcenia.cut_photometry_session(df_nph=df_nph, df_trials=df_trials, time_to_cut=10) 


#%%
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_nph = kcenia.LedState_or_Flags(df_nph)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = kcenia.verify_length(df_nph)
""" 4.1.2.2 Verify if there are repeated flags """ 
kcenia.verify_repetitions(df_nph["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
# session_day=rec.date
# plot_outliers(df_470,df_415,region,mouse,session_day) 

df_ph_1 = df_nph

# Remove rows with LedState 1 at both ends if present
if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

# Plot the data
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(df_470[region], c='#279F95', linewidth=0.5)
plt.plot(df_415[region], c='#803896', linewidth=0.5)
plt.title("Cropped signal "+subject+' '+str(session_date))
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show(block=False)
plt.close() 
# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

df_nph = df_ph_1.reset_index(drop=True)  
df_470 = df_nph[df_nph.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_nph[df_nph.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
time_diffs = (df_470["Timestamp"]).diff().dropna() 
fs = 1 / time_diffs.median() 

raw_reference = df_415[region] #isosbestic 
raw_signal = df_470[region] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = tbpod
raw_TTL_nph = tph

my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

df_nph = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium']) #IMPORTANT DF 
df_nph['times_m'] = df_nph['times'] / 60 #from seconds to minutes 

df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"].values, fs=fs) 
df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'].values, fs=fs)

column_name = ["calcium_mad", "isosbestic_mad"]
for name in column_name: 
    plt.figure(figsize=(20, 6))
    plt.plot(df_nph['times'], df_nph[name], linewidth=1.15, alpha=0.75, color='teal', label=name)
    for feedback_time in df_trials['feedback_times']:
            plt.axvline(x=feedback_time, color='gray', linewidth=1, linestyle='-', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.xlim(200,1000)
    plt.title(f'{name} with Feedback Times')
    plt.show()


#%%
""" PLOTTING THE ENTIRE SIGNAL FOR ALL THE PREPROCESSING METHODS CALCIUM AND ISOSBESTIC """

# Define the list of signal names for calcium and isosbestic
calcium_columns = ['calcium_photobleach', 'calcium_jove2019', 'calcium_mad', 'calcium_alex']
isosbestic_columns = ['isosbestic_photobleach', 'isosbestic_jove2019', 'isosbestic_mad', 'isos_alex']

# Calculate the global y-axis limits for calcium and isosbestic signals
calcium_min = min([df_nph[col].min() for col in calcium_columns])
calcium_max = max([df_nph[col].max() for col in calcium_columns])
isosbestic_min = min([df_nph[col].min() for col in isosbestic_columns])
isosbestic_max = max([df_nph[col].max() for col in isosbestic_columns])

# Create subplots (4 rows, 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(20, 20))
fig.tight_layout(pad=5)

# Plotting each calcium and corresponding isosbestic in subplots
for i, (calcium, isosbestic) in enumerate(zip(calcium_columns, isosbestic_columns)):
    # Plot calcium on the left (column 0)
    axes[i, 0].plot(df_nph['times'], df_nph[calcium], linewidth=1.15, alpha=0.75, color='teal', label=calcium)
    for feedback_time in df_trials['feedback_times']:
        axes[i, 0].axvline(x=feedback_time, color='gray', linewidth=1, linestyle='-', alpha=0.6)
    axes[i, 0].set_xlabel('Time')
    axes[i, 0].set_ylabel('Signal')
    axes[i, 0].set_xlim(200, 1000)
    axes[i, 0].set_ylim(calcium_min, calcium_max)  # Set the y-axis limits for calcium
    axes[i, 0].set_title(f'{calcium} with Feedback Times')

    # Plot isosbestic on the right (column 1)
    axes[i, 1].plot(df_nph['times'], df_nph[isosbestic], linewidth=1.15, alpha=0.75, color='purple', label=isosbestic)
    for feedback_time in df_trials['feedback_times']:
        axes[i, 1].axvline(x=feedback_time, color='gray', linewidth=1, linestyle='-', alpha=0.6)
    axes[i, 1].set_xlabel('Time')
    axes[i, 1].set_ylabel('Signal')
    axes[i, 1].set_xlim(200, 1000)
    axes[i, 1].set_ylim(isosbestic_min, isosbestic_max)  # Set the y-axis limits for isosbestic
    axes[i, 1].set_title(f'{isosbestic} with Feedback Times')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)
plt.show()







#%%
PERIEVENT_WINDOW = [time_bef,time_aft]
SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

psth_idx += event_idx


# photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
# photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
# photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
# photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
# np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
# # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6) 
# photometry_s_7 = df_nph.calcium_alex.values[psth_idx] 
# photometry_s_8 = df_nph.isos_alex.values[psth_idx] 

kcenia.plot_heatmap_psth(df_nph.calcium_mad,df_trials,psth_idx, EVENT, subject, session_date, region, eid)

# %%
""" PLOTS """ 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_session_and_part(df_nph, df_trials, xlim_min, xlim_max, preprocessingmethod): 
    # Filter the dataset based on xlim
    mask = (df_nph['times'] >= xlim_min) & (df_nph['times'] <= xlim_max)
    filtered_data = df_nph[mask]
    y_min = filtered_data[preprocessingmethod].min()
    y_max = filtered_data[preprocessingmethod].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_nph.times, df_nph[preprocessingmethod], linewidth=0.1, alpha=0.75, color='#177e89', label=preprocessingmethod)
    for feedback_time in df_trials['feedback_times']:
        ax.axvline(x=feedback_time, color='gray', linewidth=1, linestyle='-', alpha=0.6) 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(preprocessingmethod)
    # Add zoomed-in inset
    axins = inset_axes(ax, width="50%", height="50%", loc="upper right")
    axins.plot(df_nph.times, df_nph[preprocessingmethod], linewidth=0.25, alpha=0.75, color='#177e89')
    axins.set_xlim(xlim_min, xlim_max)
    axins.set_ylim(y_min, y_max)
    # Plot the feedback times in the zoomed-in inset
    for feedback_time in df_trials['feedback_times']:
        if xlim_min <= feedback_time <= xlim_max:  # Only show feedback lines within the zoomed range
            axins.axvline(x=feedback_time, color='gray', linewidth=1.05, linestyle='-', alpha=1)
    # Add rectangle around zoomed-in area in the main plot
    ax.indicate_inset_zoom(axins, edgecolor="black",)
    plt.tight_layout()
    plt.show() 

plot_session_and_part(df_nph=df_nph, df_trials=df_trials, xlim_min=1105, xlim_max=1250, preprocessingmethod='raw_calcium')
plot_session_and_part(df_nph=df_nph, df_trials=df_trials, xlim_min=1105, xlim_max=1250, preprocessingmethod='calcium_mad')
plot_session_and_part(df_nph=df_nph, df_trials=df_trials, xlim_min=1105, xlim_max=1250, preprocessingmethod='calcium_jove2019')
plot_session_and_part(df_nph=df_nph, df_trials=df_trials, xlim_min=1105, xlim_max=1250, preprocessingmethod='calcium_photobleach')
plot_session_and_part(df_nph=df_nph, df_trials=df_trials, xlim_min=1105, xlim_max=1250, preprocessingmethod='calcium_alex')



# %% 













def smooth_signal1(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.
    region_number = "4" #the ROI number you recorded from 
    nph_bnc = 0 #or 1, the BNC input you use to sync the data; "Input0" or "Input1" 
    #choose one: .csv or .pqt
    nph_file_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-06275/2023-09-18/001/raw_photometry_data/raw_photometry.csv' 
    df_nph = pd.read_csv(nph_file_path) 
    # df_nph = pd.read_parquet(nph_file_path) 
        
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
    the smoothed signal        
    """

    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

     


df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"].values, fs=fs) 
df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'].values, fs=fs)



smooth_win = 1
smooth_reference = smooth_signal1(df_nph.raw_isosbestic, smooth_win)
smooth_signal = smooth_signal1(df_nph.raw_calcium, smooth_win)



lambd = 5e4 # Adjust lambda to get the best fit
porder = 1
itermax = 50
r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
s_base=airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)
     

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1)
ax1.plot(s_base,'black',linewidth=1)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1)
ax2.plot(r_base,'black',linewidth=1)




reference = (smooth_reference - r_base)
signal = (smooth_signal - s_base)  
     

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(signal,'blue',linewidth=1)
ax2 = fig.add_subplot(212)
ax2.plot(reference,'purple',linewidth=1)
     

df_nph["calcium_test"] = signal 
df_nph["isosbestic_test"] = reference 






# %%

baseline_values = []
normalized_values = []

# Loop through each trial in df_trials
for idx in range(1,len(df_trials.intervals_0)):  # Loop through trials 

    # Get the start and end of the baseline window (next trial start and this trial end)
    baseline_start = df_trials['intervals_1'].iloc[idx-1]  # end of previous trial
    baseline_end = df_trials['intervals_0'].iloc[idx]  # start of current trial

    # Get the start and end of the trial window (this trial's start and end)
    trial_start = df_trials['intervals_0'].iloc[idx]  # start of current trial
    trial_end = df_trials['intervals_1'].iloc[idx]    # end of current trial

    # Select df_nph rows within the baseline window
    baseline_mask = (df_nph['times'] >= baseline_start) & (df_nph['times'] <= baseline_end)
    baseline_data = df_nph[baseline_mask]['calcium_test']
    
    # # Select df_nph rows within the trial window
    # trial_mask = (df_nph['times'] >= trial_start) & (df_nph['times'] <= trial_end)
    trial_data = df_nph['calcium_test']
    
    # Compute the average baseline for this trial
    avg_baseline = baseline_data.mean()
    baseline_values.append(avg_baseline)  # Store baseline values for each trial
    
    # Normalize the trial data by subtracting the baseline
    normalized_trial = trial_data - avg_baseline
    normalized_values.append(normalized_trial)  
    plt.plot()



plt.plot(normalized_trial) 
plt.show()

# %%
    # Filter the dataset based on xlim
    preprocessingmethod = "calcium_test"
    xlim_min=1105
    xlim_max=1250
    mask = (df_nph['times'] >= xlim_min) & (df_nph['times'] <= xlim_max)
    filtered_data = df_nph[mask]
    y_min = filtered_data[preprocessingmethod].min()
    y_max = filtered_data[preprocessingmethod].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_nph.times, df_nph[preprocessingmethod], linewidth=0.5, alpha=0.75, color='#177e89', label=preprocessingmethod)
    for feedback_time in df_trials['feedback_times']:
        ax.axvline(x=feedback_time, color='gray', linewidth=1, linestyle='-', alpha=0.6) 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(preprocessingmethod)
    # Add zoomed-in inset
    axins = inset_axes(ax, width="50%", height="50%", loc="upper right")
    axins.plot(df_nph.times, df_nph[preprocessingmethod], linewidth=0.75, alpha=0.75, color='#177e89')
    axins.set_xlim(xlim_min, xlim_max)
    axins.set_ylim(y_min, y_max)
    # Plot the feedback times in the zoomed-in inset
    for feedback_time in df_trials['feedback_times']:
        if xlim_min <= feedback_time <= xlim_max:  # Only show feedback lines within the zoomed range
            axins.axvline(x=feedback_time, color='gray', linewidth=1.05, linestyle='-', alpha=1)
    # Add rectangle around zoomed-in area in the main plot
    ax.indicate_inset_zoom(axins, edgecolor="black",)
    plt.tight_layout()
    plt.show() 

# %%
PERIEVENT_WINDOW = [-1,2]
SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

psth_idx += event_idx

kcenia.plot_heatmap_psth(df_nph.calcium_mad,df_trials,psth_idx, EVENT, subject, session_date, region, eid)
kcenia.plot_heatmap_psth(df_nph.calcium_test,df_trials,psth_idx, EVENT, subject, session_date, region, eid) 





#%%
""" CC CE EC EE """ 
photometry_feedback = df_nph.calcium_jove2019.values[psth_idx] 

photometry_feedback_avg = np.mean(photometry_feedback, axis=1)

import numpy as np
import matplotlib.pyplot as plt

# Function to compute avg and sem
def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

prev_feedback = df_trials['feedbackType'].shift(-1)
prev_feedback2 = df_trials['feedbackType'].shift(-2)

# Define the trial types for three consecutive feedback types
df_trials_ccc = df_trials[(prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1)]
df_trials_cce = df_trials[(prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1)]
df_trials_cec = df_trials[(prev_feedback2 == 1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1)]
df_trials_cee = df_trials[(prev_feedback2 == 1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1)]
df_trials_ecc = df_trials[(prev_feedback2 == -1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1)]
df_trials_ece = df_trials[(prev_feedback2 == -1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1)]
df_trials_eec = df_trials[(prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1)]
df_trials_eee = df_trials[(prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1)]

# PSTH for each trial type
psth_ccc = photometry_feedback[:, ((prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1))]
psth_cce = photometry_feedback[:, ((prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1))]
psth_cec = photometry_feedback[:, ((prev_feedback2 == 1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1))]
psth_cee = photometry_feedback[:, ((prev_feedback2 == 1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1))]
psth_ecc = photometry_feedback[:, ((prev_feedback2 == -1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1))]
psth_ece = photometry_feedback[:, ((prev_feedback2 == -1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1))]
psth_eec = photometry_feedback[:, ((prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1))]
psth_eee = photometry_feedback[:, ((prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1))]

# Compute avg and sem for each trial type
avg_ccc, sem_ccc = avg_sem(psth_ccc)
avg_cce, sem_cce = avg_sem(psth_cce)
avg_cec, sem_cec = avg_sem(psth_cec)
avg_cee, sem_cee = avg_sem(psth_cee)
avg_ecc, sem_ecc = avg_sem(psth_ecc)
avg_ece, sem_ece = avg_sem(psth_ece)
avg_eec, sem_eec = avg_sem(psth_eec)
avg_eee, sem_eee = avg_sem(psth_eee)

# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

# Plot for "ccc" (cold color)
color = "#218380"  # greenish blue
plt.plot(avg_ccc, color=color, linewidth=2, label='ccc trials')
plt.fill_between(range(len(avg_ccc)), avg_ccc - sem_ccc, avg_ccc + sem_ccc, color=color, alpha=0.18)

# Plot for "cce" (warm color)
color = "#ff7f0e"  # orange
plt.plot(avg_cce, color=color, linewidth=2, label="cce trials")
plt.fill_between(range(len(avg_cce)), avg_cce - sem_cce, avg_cce + sem_cce, color=color, alpha=0.18)

# Plot for "cec" (cold color)
color = "#17becf"  # blueish green
plt.plot(avg_cec, color=color, linewidth=2, label="cec trials")
plt.fill_between(range(len(avg_cec)), avg_cec - sem_cec, avg_cec + sem_cec, color=color, alpha=0.18)

# Plot for "cee" (warm color)
color = "#d62728"  # red
plt.plot(avg_cee, color=color, linewidth=2, label="cee trials")
plt.fill_between(range(len(avg_cee)), avg_cee - sem_cee, avg_cee + sem_cee, color=color, alpha=0.18)

# Plot for "ecc" (cold color)
color = "#1f77b4"  # blue
plt.plot(avg_ecc, color=color, linewidth=2, label="ecc trials")
plt.fill_between(range(len(avg_ecc)), avg_ecc - sem_ecc, avg_ecc + sem_ecc, color=color, alpha=0.18)

# Plot for "ece" (warm color)
color = "#bcbd22"  # yellow-green
plt.plot(avg_ece, color=color, linewidth=2, label="ece trials")
plt.fill_between(range(len(avg_ece)), avg_ece - sem_ece, avg_ece + sem_ece, color=color, alpha=0.18)

# Plot for "eec" (cold color)
color = "#2ca02c"  # green
plt.plot(avg_eec, color=color, linewidth=2, label="eec trials")
plt.fill_between(range(len(avg_eec)), avg_eec - sem_eec, avg_eec + sem_eec, color=color, alpha=0.18)

# Plot for "eee" (warm color)
color = "#ff9896"  # light red
plt.plot(avg_eee, color=color, linewidth=2, label="eee trials")
plt.fill_between(range(len(avg_eee)), avg_eee - sem_eee, avg_eee + sem_eee, color=color, alpha=0.18)

# Adding a vertical line, labels, title, and legend
plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.ylabel('Average Value')
plt.xlabel('Time')
title = f"psth aligned to {EVENT} "
plt.title(title + ' ' + subject + ' ' + session_date + ' ' + region, fontsize=16)

# Adding legend outside the plots
plt.legend(fontsize=14)
fig.suptitle('Neuromodulator activity for different trial types in 1 mouse', y=1.02, fontsize=18)
plt.tight_layout()

# Show the plot
plt.show()



#%%
""" CCCC, CCCE, ..., EEEE """
# Shift feedbackType column to get the previous three feedback types

photometry_feedback = df_nph.calcium_jove2019.values[psth_idx] 

photometry_feedback_avg = np.mean(photometry_feedback, axis=1)
# plt.plot(photometry_feedback_avg) 


prev_feedback = df_trials['feedbackType'].shift(-1)
prev_feedback2 = df_trials['feedbackType'].shift(-2)
prev_feedback3 = df_trials['feedbackType'].shift(-3)

# Define the trial types for four consecutive feedback types
df_trials_cccc = df_trials[(prev_feedback3 == 1) & (prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1)]
df_trials_ccce = df_trials[(prev_feedback3 == 1) & (prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1)]
df_trials_eeec = df_trials[(prev_feedback3 == -1) & (prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1)]
df_trials_eeee = df_trials[(prev_feedback3 == -1) & (prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1)]

# PSTH for each trial type
psth_cccc = photometry_feedback[:, ((prev_feedback3 == 1) & (prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == 1))]
psth_ccce = photometry_feedback[:, ((prev_feedback3 == 1) & (prev_feedback2 == 1) & (prev_feedback == 1) & (df_trials['feedbackType'] == -1))]
psth_eeec = photometry_feedback[:, ((prev_feedback3 == -1) & (prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == 1))]
psth_eeee = photometry_feedback[:, ((prev_feedback3 == -1) & (prev_feedback2 == -1) & (prev_feedback == -1) & (df_trials['feedbackType'] == -1))]

# Compute avg and sem for each trial type
avg_cccc, sem_cccc = avg_sem(psth_cccc)
avg_ccce, sem_ccce = avg_sem(psth_ccce)
avg_eeec, sem_eeec = avg_sem(psth_eeec)
avg_eeee, sem_eeee = avg_sem(psth_eeee)

# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

# Plot for "cccc" (cold color)
color = "#1f77b4"  # blue
plt.plot(avg_cccc, color=color, linewidth=2, label='cccc trials')
plt.fill_between(range(len(avg_cccc)), avg_cccc - sem_cccc, avg_cccc + sem_cccc, color=color, alpha=0.18)

# Plot for "ccce" (warm color)
color = "#ff7f0e"  # orange
plt.plot(avg_ccce, color=color, linewidth=2, label="ccce trials")
plt.fill_between(range(len(avg_ccce)), avg_ccce - sem_ccce, avg_ccce + sem_ccce, color=color, alpha=0.18)

# Plot for "eeec" (cold color)
color = "#2ca02c"  # green
plt.plot(avg_eeec, color=color, linewidth=2, label="eeec trials")
plt.fill_between(range(len(avg_eeec)), avg_eeec - sem_eeec, avg_eeec + sem_eeec, color=color, alpha=0.18)

# Plot for "eeee" (warm color)
color = "#d62728"  # red
plt.plot(avg_eeee, color=color, linewidth=2, label="eeee trials")
plt.fill_between(range(len(avg_eeee)), avg_eeee - sem_eeee, avg_eeee + sem_eeee, color=color, alpha=0.18)

# Adding a vertical line, labels, title, and legend
plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.ylabel('Average Value')
plt.xlabel('Time')
title = f"psth aligned to {EVENT} "
plt.title(title + ' ' + subject + ' ' + session_date + ' ' + region, fontsize=16)

# Adding legend outside the plots
plt.legend(fontsize=14)
fig.suptitle('Neuromodulator activity for different trial types in 1 mouse', y=1.02, fontsize=18)
plt.tight_layout()

# Show the plot
plt.show()



# %%
""" BLOCK SWITCHES """ 
import numpy as np
import matplotlib.pyplot as plt

# Assuming df_trials is already defined and populated

# Extract relevant data from df_trials
probabilities = df_trials['probabilityLeft'].values
feedback_types = df_trials['feedbackType'].values

# Identify indices for bl and br conditions
bl_indices = (np.roll(probabilities, -1) == 0.5) | (np.roll(probabilities, -1) == 0.2)  # Previous was 0.5 or 0.2
br_indices = (np.roll(probabilities, -1) == 0.5) | (np.roll(probabilities, -1) == 0.8)  # Previous was 0.5 or 0.8
bl_indices = bl_indices & (probabilities == 0.8)  # Current is 0.8
br_indices = br_indices & (probabilities == 0.2)  # Current is 0.2

# Get the feedback types for correct (1) and incorrect (-1) responses
correct_bl = bl_indices & (feedback_types == 1)
incorrect_bl = bl_indices & (feedback_types == -1)
correct_br = br_indices & (feedback_types == 1)
incorrect_br = br_indices & (feedback_types == -1)

# PSTH for each condition
psth_bl_correct = photometry_feedback[:, correct_bl]
psth_bl_incorrect = photometry_feedback[:, incorrect_bl]
psth_br_correct = photometry_feedback[:, correct_br]
psth_br_incorrect = photometry_feedback[:, incorrect_br]

# Function to compute average and SEM
def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

# Compute average and SEM for each condition
avg_bl_correct, sem_bl_correct = avg_sem(psth_bl_correct)
avg_bl_incorrect, sem_bl_incorrect = avg_sem(psth_bl_incorrect)
avg_br_correct, sem_br_correct = avg_sem(psth_br_correct)
avg_br_incorrect, sem_br_incorrect = avg_sem(psth_br_incorrect)

# Create the figure
plt.figure(figsize=(12, 12))

# Plot for bl correct trials
color = "#1f77b4"  # Blue
plt.plot(avg_bl_correct, color=color, linewidth=2, label='bl Correct')
plt.fill_between(range(len(avg_bl_correct)), avg_bl_correct - sem_bl_correct, avg_bl_correct + sem_bl_correct, color=color, alpha=0.18)

# Plot for bl incorrect trials
color = "#ff7f0e"  # Orange
plt.plot(avg_bl_incorrect, color=color, linewidth=2, label='bl Incorrect')
plt.fill_between(range(len(avg_bl_incorrect)), avg_bl_incorrect - sem_bl_incorrect, avg_bl_incorrect + sem_bl_incorrect, color=color, alpha=0.18)

# Plot for br correct trials
color = "#2ca02c"  # Green
plt.plot(avg_br_correct, color=color, linewidth=2, label='br Correct')
plt.fill_between(range(len(avg_br_correct)), avg_br_correct - sem_br_correct, avg_br_correct + sem_br_correct, color=color, alpha=0.18)

# Plot for br incorrect trials
color = "#d62728"  # Red
plt.plot(avg_br_incorrect, color=color, linewidth=2, label='br Incorrect')
plt.fill_between(range(len(avg_br_incorrect)), avg_br_incorrect - sem_br_incorrect, avg_br_incorrect + sem_br_incorrect, color=color, alpha=0.18)

# Adding a vertical line, labels, title, and legend
plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.ylabel('Average Value')
plt.xlabel('Time')
plt.title(f'PSTH Aligned to {EVENT}', fontsize=16)

# Adding legend outside the plots
plt.legend(fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

# %%
""" CHOICES and CORRECT ERROR""" 
import numpy as np
import matplotlib.pyplot as plt

# Function to compute avg and sem
def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

# Separate the trials based on choice
df_trials_cl = df_trials[df_trials['choice'] == 1]  # cl: choice = 1
df_trials_cr = df_trials[df_trials['choice'] == -1]  # cr: choice = -1
df_trials_cn = df_trials[df_trials['choice'] == 0]   # cn: choice = 0

# Filter by feedbackType (correct and incorrect)
# For cl
cl_indices_correct = df_trials_cl.index[df_trials_cl['feedbackType'] == 1].tolist()
cl_indices_incorrect = df_trials_cl.index[df_trials_cl['feedbackType'] == -1].tolist()

# For cr
cr_indices_correct = df_trials_cr.index[df_trials_cr['feedbackType'] == 1].tolist()
cr_indices_incorrect = df_trials_cr.index[df_trials_cr['feedbackType'] == -1].tolist()

# For cn
cn_indices_correct = df_trials_cn.index[df_trials_cn['feedbackType'] == 1].tolist()
cn_indices_incorrect = df_trials_cn.index[df_trials_cn['feedbackType'] == -1].tolist()

# PSTH for each trial type using the original indices
psth_cl_correct = photometry_feedback[:, cl_indices_correct]
psth_cl_incorrect = photometry_feedback[:, cl_indices_incorrect]

psth_cr_correct = photometry_feedback[:, cr_indices_correct]
psth_cr_incorrect = photometry_feedback[:, cr_indices_incorrect]

psth_cn_correct = photometry_feedback[:, cn_indices_correct]
psth_cn_incorrect = photometry_feedback[:, cn_indices_incorrect]

# Compute avg and sem for each trial type
avg_cl_correct, sem_cl_correct = avg_sem(psth_cl_correct)
avg_cl_incorrect, sem_cl_incorrect = avg_sem(psth_cl_incorrect)

avg_cr_correct, sem_cr_correct = avg_sem(psth_cr_correct)
avg_cr_incorrect, sem_cr_incorrect = avg_sem(psth_cr_incorrect)

avg_cn_correct, sem_cn_correct = avg_sem(psth_cn_correct)
avg_cn_incorrect, sem_cn_incorrect = avg_sem(psth_cn_incorrect)

# Create the figure
plt.figure(figsize=(12, 12))

# Plot for "cl" correct trials
color = "#1f77b4"  # Blue
plt.plot(avg_cl_correct, color=color, linewidth=2, label='cl Correct')
plt.fill_between(range(len(avg_cl_correct)), avg_cl_correct - sem_cl_correct, avg_cl_correct + sem_cl_correct, color=color, alpha=0.18)

# Plot for "cl" incorrect trials
color = "#ff7f0e"  # Orange
plt.plot(avg_cl_incorrect, color=color, linewidth=2, label='cl Incorrect')
plt.fill_between(range(len(avg_cl_incorrect)), avg_cl_incorrect - sem_cl_incorrect, avg_cl_incorrect + sem_cl_incorrect, color=color, alpha=0.18)

# Plot for "cr" correct trials
color = "#2ca02c"  # Green
plt.plot(avg_cr_correct, color=color, linewidth=2, label='cr Correct')
plt.fill_between(range(len(avg_cr_correct)), avg_cr_correct - sem_cr_correct, avg_cr_correct + sem_cr_correct, color=color, alpha=0.18)

# Plot for "cr" incorrect trials
color = "#d62728"  # Red
plt.plot(avg_cr_incorrect, color=color, linewidth=2, label='cr Incorrect')
plt.fill_between(range(len(avg_cr_incorrect)), avg_cr_incorrect - sem_cr_incorrect, avg_cr_incorrect + sem_cr_incorrect, color=color, alpha=0.18)

# Plot for "cn" correct trials
color = "#bcbd22"  # Yellow-green
plt.plot(avg_cn_correct, color=color, linewidth=2, label='cn Correct')
plt.fill_between(range(len(avg_cn_correct)), avg_cn_correct - sem_cn_correct, avg_cn_correct + sem_cn_correct, color=color, alpha=0.18)

# Plot for "cn" incorrect trials
color = "#ff9896"  # Light red
plt.plot(avg_cn_incorrect, color=color, linewidth=2, label='cn Incorrect')
plt.fill_between(range(len(avg_cn_incorrect)), avg_cn_incorrect - sem_cn_incorrect, avg_cn_incorrect + sem_cn_incorrect, color=color, alpha=0.18)

# Adding a vertical line, labels, title, and legend
plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.ylabel('Average Value')
plt.xlabel('Time')
plt.suptitle(f'PSTH Aligned to {EVENT}', fontsize=16) 
plt.title(title + ' ' + subject + ' ' + session_date + ' ' + region, fontsize=16)

# Adding legend outside the plots
plt.legend(fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

#%% 
""" VIDEO """
test = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'])
video_data = pd.DataFrame(test['lightningPose']) 
video_data["times"] = test.times 
#%%
""" 1. try to recalculate the diameter and correlate it with 5-HT """ 
video_data["v0_diameter"] = video_data["pupil_top_r_y"] - video_data["pupil_bottom_r_y"] 
video_data["v1_diameter"] = ((video_data.pupil_top_r_y - video_data.pupil_bottom_r_y) + (video_data.pupil_left_r_y - video_data.pupil_right_r_y))/2

""" 2. Nose Tip """ 
video_data["nose_s"] = (video_data.nose_tip_x + video_data.nose_tip_y) 
video_data["nose_m"] = (video_data.nose_tip_x * video_data.nose_tip_y) 

""" 3. Paws """
video_data["paw_l_s"] = (video_data.paw_l_x + video_data.paw_l_y) 
video_data["paw_l_m"] = (video_data.paw_l_x * video_data.paw_l_y) 
video_data["paw_r_s"] = (video_data.paw_r_x + video_data.paw_r_y) 
video_data["paw_r_m"] = (video_data.paw_r_x * video_data.paw_r_y) 
video_data["paw_lr_s"] = ((video_data.paw_l_x + video_data.paw_l_y) + (video_data.paw_r_x + video_data.paw_r_y)) 
video_data["paw_lr_m"] = ((video_data.paw_l_x * video_data.paw_l_y) + (video_data.paw_r_x * video_data.paw_r_y)) 
video_data["paw_lr_y_s"] = (video_data.paw_l_y + video_data.paw_r_y) 
video_data["paw_lr_y_m"] = (video_data.paw_l_y * video_data.paw_r_y)




# %%
""" CORRELATIONS BETWEEN VIDEO DATA AND PREPROCESSING METHODS""" 
# Define window size for rolling mean
window_size = 10

# List of video column names
video_column_names = video_data.columns

# Function to plot correlations for a given calcium variable
def plot_correlations(calcium_variable_name, calcium_variable):
    # Initialize a DataFrame to hold correlation data
    correlation_results = {}

    # Loop through video variable columns to calculate correlations
    for video_variable in video_column_names:
        # Create smoothed data
        df_nph_smoothed = df_nph[calcium_variable].rolling(window=window_size).mean()
        video_data_smoothed = video_data[video_variable].rolling(window=window_size).mean()

        # Combine into a DataFrame and drop NaN values
        combined_data = pd.DataFrame({
            calcium_variable_name: df_nph_smoothed,
            video_variable: video_data_smoothed
        }).dropna()

        # Calculate correlation and store the results
        correlation = combined_data.corr().iloc[0, 1]  # Get correlation between calcium and video variable
        correlation_results[video_variable] = correlation

    # Convert results to DataFrame for plotting
    correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index', columns=['Correlation'])
    correlation_df = correlation_df.reset_index().rename(columns={'index': 'Video Variable'})

    # Plotting the correlation results
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Correlation', y='Video Variable', data=correlation_df, palette='coolwarm')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f'Correlation of {calcium_variable_name} with Video Variables')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Video Variable')
    plt.xlim(-0.05, 0.05)
    plt.grid(True)
    plt.show()

# Plot correlations for both calcium variables
plot_correlations('Calcium (mad)', 'calcium_mad')
plot_correlations('Calcium (jove 2019)', 'calcium_jove2019') 




#%%
""" WHEEL DATA """ 

""" WHEEL MOVEMENT - shadow areas are wheel movement """ 
wheel = one.load_object(eid, 'wheel', collection='alf')
try:
    # Warning: Some older sessions may not have a wheelMoves dataset
    wheel_moves = one.load_object(eid, 'wheelMoves', collection='alf')
except AssertionError:
    wheel_moves = extract_wheel_moves(wheel.timestamps, wheel.position) 

fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 10
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
plt.legend()

nph_smoothed = df_nph["calcium_jove2019"].rolling(window=window_size).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='NM')

# Add vertical lines for feedback times
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=1)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=1)

# Add shaded areas for wheel movements
for interval in wheel_moves['intervals']:
    ax1.axvspan(interval[0], interval[1], color='gray', alpha=0.5)

# ax2.set_ylim(-0.002, 0.002)
plt.legend()
plt.xlim(1050, 1150)
plt.show()
# %%
