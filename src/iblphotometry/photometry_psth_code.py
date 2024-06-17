# %% 
########################################################################
# KB 2024-06-17  
# 
# loop in order to also have the behavior and the psth_idx tables saved 
# from load_processed_data.py from KceniaB GitHub 
######################################################################## 

import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from one.api import ONE
one = ONE() 


#%%
# READ NPH DATA 
""" CHANGE HERE """ 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
prefix = 'demux_' 
file_name = 'demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt'
# psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

mouse = file_name[10:19]
date = file_name[20:30] 
region_number = file_name[31:32] 
print(mouse, date, region_number)

eid,df_trials = get_eid(mouse,date) 
df_nph = nphca_1

# create trialNumber
df_trials['trialNumber'] = range(1, len(df_trials) + 1)

# create allContrasts 
idx = 2
new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
# create allSContrasts 
df_trials['allSContrasts'] = df_trials['allContrasts']
df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

# # create reactionTime
# reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
# df_trials["reactionTime"] = reactionTime 

# add session info 
df_trials["mouse"] = mouse
df_trials["date"] = date 
df_trials["regionNumber"] = region_number
df_trials["eid"] = eid 
# df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

# remove all trials that are not totally associated with photometry, 2 seconds after the photometry was turned off 
while (df_trials["intervals_1"].iloc[-1] + 62) >= df_nph["times"].iloc[-1]:
    df_trials = df_trials.iloc[:-1]

# SAVE THE BEHAVIOR TABLE 
# df_trials.to_parquet(f'/home/kceniabougrova/Documents/results_for_OW/trialstable_{mouse}_{date}_{region_number}_{eid}.pqt') 

array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
# print(idx_event) 

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

photometry_feedback = df_nph.calcium.values[psth_idx] 

# np.save(f'/home/kceniabougrova/Documents/results_for_OW/psthidx_{EVENT_NAME}_{mouse}_{date}_{region_number}_{eid}.npy', photometry_feedback) 

sns.heatmap(photometry_feedback)