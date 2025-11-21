# %%
from brainbox.io.one import PhotometrySessionLoader
from tqdm import tqdm
from iblphotometry.pipelines import sliding_mad_pipeline, run_pipeline
import numpy as np
import pynapple as nap
import pandas as pd
import matplotlib.pyplot as plt
from one.api import ONE
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from trial_type_definitions import add_info_to_trials_table, event_definitions_biasedCW

# %% get sessions for subject
one = ONE()
# subject = 'ZFM-08776'
# subject = 'ZFM-08818'
subject = 'ZFM-08828'

django = [
    f'subject__nickname,{subject}',
    # 'start_time__lt,2025-10-20',
    'task_protocol__icontains,biased',
]
sessions = one.alyx.rest('sessions', 'list', django=django)
sessions = sessions
print(len(sessions))

# %% only extracted sessions

loaders = {}
for session in tqdm(sessions):
    if 'alf/photometry' in one.list_collections(session['id']):
        loaders[session['id']] = PhotometrySessionLoader(eid=session['id'], one=one)
        loaders[session['id']].load_photometry()

sessions = [s for s in sessions if s['id'] in loaders.keys()]
print(len(loaders))

# %%
MIN_TRIALS = 10  # per analysis

# aggregate over sessions
aggregate_signals = pd.DataFrame(columns=['eid', 'analysis', 'value'])
trials_dfs = {}

for eid in loaders.keys():
    signal = loaders[eid].photometry['GCaMP']['VTA']
    signal = run_pipeline(sliding_mad_pipeline, signal)
    signal = nap.Tsd(signal.index, signal.values)

    trials_df = loaders[eid].trials
    trials_df = add_info_to_trials_table(trials_df)

    for analysis, definition in event_definitions_biasedCW.items():
        # the subselection of trials
        _trials_df = trials_df.query(definition['query'])
        timestamps = _trials_df[definition['align_event']].values
        # drop invalid
        timestamps = timestamps[~pd.isna(timestamps)]

        frame = nap.compute_perievent_continuous(signal, nap.Ts(timestamps), minmax=definition['window'])

        # time and trial average
        if timestamps.shape[0] > MIN_TRIALS:  # exclude trial types with too little trials
            value = np.average(frame, axis=(0, 1))
        else:
            value = np.nan
        v = pd.DataFrame([dict(eid=eid, analysis=analysis, value=value)])
        aggregate_signals = pd.concat([aggregate_signals, v], ignore_index=True)

# %% merge aggregate signals with session info for easier plotting
for eid, group in aggregate_signals.groupby('eid'):
    (session,) = list(filter(lambda s: s['id'] == eid, sessions))
    aggregate_signals.loc[group.index, 'start_time'] = session['start_time']

aggregate_signals = aggregate_signals.sort_values(by='start_time')

# %% plot aggregate signals
analyses = event_definitions_biasedCW.keys()  # plot all
# analyses = ['fback0','fback1']

colors = dict(
    zip(
        analyses,
        sns.color_palette('husl', n_colors=len(analyses)),
    ),
)

fig, axes = plt.subplots()

for analysis in analyses:
    try:
        group = aggregate_signals.groupby('analysis').get_group(analysis)
        axes.plot(
            group['start_time'].astype('datetime64[ns]'),
            group['value'],
            '.',
            color=colors[analysis],
        )
        axes.plot(
            group['start_time'].astype('datetime64[ns]'),
            group['value'],
            lw=1,
            alpha=0.5,
            color=colors[analysis],
            label=analysis,
        )
    except KeyError:
        pass

axes.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
axes.xaxis.set_major_locator(mdates.DayLocator())
axes.legend()
sns.despine(fig)
axes.set_title(subject)
fig.autofmt_xdate()


# %%
