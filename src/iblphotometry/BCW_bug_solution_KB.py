""" 
KB 2025-05-25
Solution proposed on Oct2024

Code to fix the BCW bug in the IBL photometry sessions

This code is used to fix the bug in the labelling of some of the photometry BCW sessions - the task ran correctly, but the 
labelling was incorrect for the probabilityLeft: 
 - BCW block 1 - 50/50 = correct
 - BCW block 2 - 20/80 or 80/20 = correct
 - BCW block 3 - 80/20 or 20/80 = incorrect - it starts shifting 0.2 and 0.8 for each consecutive trial
 
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
from one.api import ONE #always after the imports 
one = ONE(cache_rest=None, mode='remote') 

#functions 
def get_eid(): 
    # eids = one.search(project='ibl_fibrephotometry') 
    # use example eid with bug
    eid = 'd3ac8f32-cfba-4f48-b513-9d998ab0ae72'
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    try:
        # Try to load the trials directly
        a = one.load_object(eid, 'trials')
        trials = a.to_df()
        trials['trialNumber'] = range(1, len(trials) + 1) 
        trials["mouse"] = ref.subject
        trials["date"] = ref.date
        trials["eid"] = eid 
        df_trials = trials
    except: 
        print("session not found")

    return df_trials, eid

df_trials, eid = get_eid()

plt.plot(df_trials.probabilityLeft, alpha=0.5)


""" useful""" 
# eids = one.search(project='ibl_fibrephotometry') 

#%%
""" LOAD TRIALS """
def load_trials_updated(eid=eid): 
    trials = one.load_object(eid, 'trials')
    ref = one.eid2ref(eid)
    subject = ref.subject
    session_date = str(ref.date) 
    if len(trials['intervals'].shape) == 2: 
        trials['intervals_0'] = trials['intervals'][:, 0]
        trials['intervals_1'] = trials['intervals'][:, 1]
        del trials['intervals']  # Remove original nested array 
    df_trials = pd.DataFrame(trials) 
    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    # create allSContrasts 
    df_trials['allSContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
    df_trials[["subject", "date", "eid"]] = [subject, session_date, eid]    
    df_trials["reactionTime"] = df_trials["firstMovement_times"] - df_trials["stimOnTrigger_times"]
    df_trials["responseTime"] = df_trials["response_times"] - df_trials["stimOnTrigger_times"] 
    df_trials["quiescenceTime"] = df_trials["stimOnTrigger_times"] - df_trials["intervals_0"] 
    df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  

    try: 
        dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
        values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
        # values gives the block length 
        # example for     eid = 'd3ac8f32-cfba-4f48-b513-9d998ab0ae72'
        # values = [90, 41, 65, 62, 30, 31, 64, 82, 33, 80, 70, 62, 40, 70, 22, 72, 60, 22, 30, 53, 51, 28, 31, 28, 22, 41, 72, 72, 51, 70, 24, 30, 55, 78, 39, 53, 23, 53, 25, 21, 48, 97]


        values_sum = np.cumsum(values) 

        # Initialize a new column 'probL' with NaN values
        df_trials['probL'] = np.nan

        # Set the first block (first `values_sum[0]` rows) to 0.5
        df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 


        df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']

        previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 


        # Iterate over the blocks starting from values_sum[1]
        for i in range(1, len(values_sum)-1):
            print("i = ", i)
            start_idx = values_sum[i]
            end_idx = values_sum[i+1]-1
            print("start and end _idx = ", start_idx, end_idx)
            
            # Assign the block value based on the previous one
            if previous_value == 0.2:
                current_value = 0.8
            else:
                current_value = 0.2
            print("current value = ", current_value)


            # Set the 'probL' values for the current block
            df_trials.loc[start_idx:end_idx, 'probL'] = current_value
            
            # Update the previous_value for the next block
            previous_value = current_value

        # Handle any remaining rows after the last value_sum block
        if len(df_trials) > values_sum[-1]:
            df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

        # plt.plot(df_trials.probabilityLeft, alpha=0.5)
        # plt.plot(df_trials.probL, alpha=0.5)
        # plt.title(f'behavior_{subject}_{session_date}_{eid}')
        # plt.show() 
    except: 
        pass 

    df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
    return df_trials, subject, session_date

df_trials, subject, session_date = load_trials_updated(eid) 

plt.plot(df_trials.probabilityLeft, alpha=0.5) #bug
plt.plot(df_trials.probL, alpha=0.85, color='red') #bug solved

# %%
