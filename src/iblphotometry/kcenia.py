
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


