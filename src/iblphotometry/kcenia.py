
import numpy as np
import pandas as pd 
from one.api import ONE
one = ONE()
from ibllib.io.extractors.biased_trials import extract_all 
from brainbox.io.one import SessionLoader
import neurodsp.utils

# def get_regions(rec): 
#     """ 
#     extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
#     """
#     regions = [f"Region{rec.region}G"] 
#     if not np.isnan(rec.region2): 
#         regions.append(f"Region{rec.region2}G")
#     return regions 

def get_regions(rec): 
    """ 
    extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
    """
    regions = [f"Region{rec.region}G"] 
    return regions


def get_nph(rec): 
    source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/")
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    df_nphttl = pd.read_csv(source_folder+f"bonsai_DI{rec.nph_bnc}{rec.nph_file}.csv") 
    return df_nph, df_nphttl 

def get_eid(rec): 
    eids = one.search(subject=rec.mouse, date=rec.date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    try:
        # Try to load the trials directly
        a = one.load_object(eid, 'trials')
        trials = a.to_df()
    except Exception as e:
        # If loading fails, use the alternative method
        print("Failed to load trials directly. Using alternative method...")
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 
    
def get_ttl(df_DI0, df_trials): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])
    return tph, tbpod 



def start_2_end_1(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=0, starting at flag=2, finishing at flag=1, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["LedState"][0] == 0: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if (array1["LedState"][0] != 2) or (array1["LedState"][0] != 1): 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][0] == 1: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["LedState"][len(array1)-1] == 2: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
def start_17_end_18(df_photometry): 
    """
    input = raw photometry data
    output = photometry dataframe without the initial flag=16, starting at flag=17, finishing at flag=18, reset_index applied 
    """
    df_photometry = df_photometry.reset_index(drop=True)
    array1 = df_photometry
    if array1["Flags"][0] == 16: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][0] == 18: 
        array1 = array1[1:len(array1)]
        array1 = array1.reset_index(drop=True)
    if array1["Flags"][len(array1)-1] == 17: 
        array1 = array1[0:len(array1)-1] 
        array1 = array1.reset_index(drop=True)
    array2 = pd.DataFrame(array1)
    return(array2) 
""" 4.1.1 Change the Flags that are combined to Flags that will represent only the LED that was on """ 
"""1 and 17 are isosbestic; 2 and 18 are GCaMP"""
def change_flags(df_with_flags): 
    df_with_flags = df_with_flags.reset_index(drop=True)
    if 'LedState' in df_with_flags.columns: 
        array1 = np.array(df_with_flags["LedState"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["LedState"] = array2
        return(df_with_flags) 
    else: 
        array1 = np.array(df_with_flags["Flags"])
        for i in range(0,len(df_with_flags)): 
            if array1[i] == 529 or array1[i] == 273 or array1[i] == 785 or array1[i] == 17: 
                array1[i] = 1
            elif array1[i] == 530 or array1[i] == 274 or array1[i] == 786 or array1[i] == 18: 
                array1[i] = 2
            else: 
                array1[i] = array1[i] 
        array2 = pd.DataFrame(array1)
        df_with_flags["Flags"] = array2
        return(df_with_flags) 
















#%%

def LedState_or_Flags(df_PhotometryData): 
    if 'LedState' in df_PhotometryData.columns:                         #newversion 
        df_PhotometryData = start_2_end_1(df_PhotometryData)
        df_PhotometryData = df_PhotometryData.reset_index(drop=True)
        df_PhotometryData = (change_flags(df_PhotometryData))
    else:                                                               #oldversion
        df_PhotometryData = start_17_end_18(df_PhotometryData) 
        df_PhotometryData = df_PhotometryData.reset_index(drop=True) 
        df_PhotometryData = (change_flags(df_PhotometryData))
        df_PhotometryData["LedState"] = df_PhotometryData["Flags"]
    return df_PhotometryData

# %%
# def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    import numpy as np
    import pandas as pd
    if one is None:
        from one.api import ONE 
        ONE() 
        one = ONE() 

    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'],
         trials['feedback_times'], trials['firstMovement_times'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times', 'trials.firstMovement_times',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.shape[0] == 0:
            return
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
#        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
#          trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
#          trials['feedbackType'], trials['choice'], trials['firstMovement_times'],
#          trials['feedback_times']) = one.load(
#                              eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
#                                                  'trials.goCue_times', 'trials.probabilityLeft',
#                                                  'trials.contrastLeft', 'trials.contrastRight',
#                                                  'trials.feedbackType', 'trials.choice',
#                                                  'trials.firstMovement_times',
#                                                  'trials.feedback_times'])
        try:
            trials = one.load_object(eid, 'trials') #210810 Updated by brandon due to ONE update
        except:
            return {}
            
            
            
    if len(trials['probabilityLeft']) == 0: # 210810 Updated by brandon due to ONE update
        return
#     if trials.shape[0] == 0:
#         return
#     trials['signed_contrast'] = trials['contrastRight']
#     trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
#     trials['correct'] = trials['feedbackType']
#     trials.loc[trials['correct'] == -1, 'correct'] = 0
#     trials['right_choice'] = -trials['choice']
#     trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
#     trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
#     trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
#                'stim_side'] = 1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
#                'stim_side'] = -1
    assert np.all(np.logical_xor(np.isnan(trials['contrastRight']),np.isnan(trials['contrastLeft'])))
    
    trials['signed_contrast'] = np.copy(trials['contrastRight'])
    use_trials = np.isnan(trials['signed_contrast'])
    trials['signed_contrast'][use_trials] = -np.copy(trials['contrastLeft'])[use_trials]
    trials['correct'] = trials['feedbackType']
    use_trials = (trials['correct'] == -1)
    trials['correct'][use_trials] = 0
    trials['right_choice'] = -np.copy(trials['choice'])
    use_trials = (trials['right_choice'] == -1)
    trials['right_choice'][use_trials] = 0
    trials['stim_side'] = (np.isnan(trials['contrastLeft'])).astype(int)
    use_trials = (trials['stim_side'] == 0)
    trials['stim_side'][use_trials] = -1
#     if 'firstMovement_times' in trials.columns.values:
    trials['reaction_times'] = np.copy(trials['firstMovement_times'] - trials['goCue_times'])
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials







def verify_length(df_PhotometryData): 
    """
    Checking if the length is different
    x = df_470
    y = df_415
    """ 
    x = df_PhotometryData[df_PhotometryData.LedState==2]
    y = df_PhotometryData[df_PhotometryData.LedState==1] 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's") 
    print("470 = ",x.LedState.count()," 415 = ",y.LedState.count())
    return(x,y)


def verify_repetitions(x): 
    """
    Checking if there are repetitions in consecutive rows
    x = df_PhotometryData["Flags"]
    """ 
    for i in range(1,(len(x)-1)): 
        if x[i-1] == x[i]: 
            print("here: ", i)



def find_FR(x): 
    """
    find the frame rate of acquisition
    x = df_470["Timestamp"]
    """
    acq_FR = round(1/np.mean(x.diff()))
    # check to make sure that it is 15/30/60! (- with a loop)
    if acq_FR == 30 or acq_FR == 60 or acq_FR == 120: 
        print("All good, the FR is: ", acq_FR)
    else: 
        print("CHECK FR!!!!!!!!!!!!!!!!!!!!") 
    return acq_FR 




















# #%%
# """ CODE BEFORE EDITING 18APR2024 """

# def extract_data_info(df): 
#     """ 
#     extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
#     """
#     for i in range(len(df["Mouse"])): 
#         mouse = df.Mouse.values[0] #"ZFM-06948" 
#         date = df['date'].dt.strftime('%Y-%m-%d').values[0] #"2024-03-22"
#         nphfile_number = str(df.nph_file.values[0]) #"0"
#         bncfile_number = str(df.nph_bnc.values[0]) #"0"
#         region = "Region"+str(df.region.values[0])+"G"
#         region2 = "Region"+str(df.region2.values[0])+"G" 
#         if mouse == "ZFM-06948" or mouse == "ZFM-06305": 
#             nm = "ACh" 
#         elif mouse == "ZFM-06275": 
#             nm = "NE" 
#     return mouse, date, nphfile_number, bncfile_number, region, region2, nm

# def get_nph(date, nphfile_number, bncfile_number): 
#     source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
#     df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
#     df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv") 
#     return df_nph, df_nphttl 

# def get_eid(mouse,date): 
#     eids = one.search(subject=mouse, date=date) 
#     eid = eids[0]
#     ref = one.eid2ref(eid)
#     print(eid)
#     print(ref) 
#     try:
#         # Try to load the trials directly
#         a = one.load_object(eid, 'trials')
#         trials = a.to_df()
#     except Exception as e:
#         # If loading fails, use the alternative method
#         print("Failed to load trials directly. Using alternative method...")
#         session_path_behav = '/home/kceniabougrova/Documents/nph/Behav_2024Mar20/ZFM-06948/2024-03-22/001/'
#         df_alldata = extract_all(session_path_behav)
#         table_data = df_alldata[0]['table']
#         trials = pd.DataFrame(table_data) 
#     return eid, trials 
    
# def get_ttl(df_DI0, df_trials): 
#     if 'Value.Value' in df_DI0.columns: #for the new ones
#         df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
#     else:
#         df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
#     #use Timestamp from this part on, for any of the files
#     raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
#     df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
#     # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
#     df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
#     tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
#     tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])
#     return tph, tbpod 


