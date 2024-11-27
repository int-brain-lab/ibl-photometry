# %%
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
import warnings
from ibllib.io.extractors.fibrephotometry import (
    NEUROPHOTOMETRICS_LED_STATES,
    LIGHT_SOURCE_MAP,
)

# def from_csv(): ...


def from_dataframe(
    raw_df: pd.DataFrame,
    data_columns: list[str] = None,
    time_column: str = None,
    channel_column: str = 'name',
    channel_names: list[str] = None,
    rename: dict = None,
):
    # from a raw dataframe as it is stored in ONE (signal.pqt)
    # data_columns is a list of str that specifies the names of the column that hold the actual data, like 'RegionXX'
    # channel_column is the column that specifies the temporally multiplexed acquisition channels

    # infer if not explicitly provided: defaults to everything that starts with 'Region'
    if data_columns is None:
        data_columns = [col for col in raw_df.columns if col.startswith('Region')]

    # infer name of time column if not provided
    if time_column is None:
        (time_column,) = [col for col in raw_df.columns if 'time' in col.lower()]

    # infer channel names if they are not explicitly provided
    if channel_names is None:
        channel_names = raw_df[channel_column].unique()

    # drop empty acquisition channels
    to_drop = ['None', '']
    channel_names = [ch for ch in channel_names if ch not in to_drop]

    raw_tfs = {}
    for channel in channel_names:
        # TODO include the use of raw_df['include'] to set the time_support of the pynapple object
        # requires conversion of boolen to nap.IntervalSet (have done this somewhere already. find code)

        # TODO check pynappe PR#343 https://github.com/pynapple-org/pynapple/pull/343 for future
        # inclusion of locations_df as metadata

        # get the data for the band
        df = raw_df.groupby(channel_column).get_group(channel)
        # if rename dict is passed, rename Region0X to the corresponding brain region
        if rename is not None:
            df = df.rename(columns=rename)
            data_columns = rename.values()
        raw_tfs[channel] = nap.TsdFrame(df.set_index(time_column)[data_columns])

    return raw_tfs


def from_pqt(
    signal_pqt_path: str | Path,
    locations_pqt_path: str | Path = None,
):
    # from .signal.pqt files as they are registered in alyx

    raw_df = pd.read_parquet(signal_pqt_path)
    if locations_pqt_path is not None:
        locations_df = pd.read_parquet(locations_pqt_path)
        data_columns = list(locations_df.index)
    else:
        warnings.warn(
            'loading a photometry.signal.pqt file without its corresponding photometryROI.locations.pqt'
        )
        data_columns = None

    read_config = dict(
        data_columns=data_columns,
        time_column='times',
        channel_column='name',
    )

    return from_dataframe(raw_df, **read_config)


def _read_raw_neurophotometrics_df(raw_df: pd.DataFrame, rois=None) -> pd.DataFrame:
    #
    if rois is None:
        rois = raw_df.columns[4:]

    out_df = raw_df.filter(items=rois, axis=1).sort_index(axis=1)
    timestamp_name = (
        'SystemTimestamp' if 'SystemTimestamp' in raw_df.columns else 'Timestamp'
    )
    out_df['times'] = raw_df[timestamp_name]
    out_df['wavelength'] = np.nan
    out_df['name'] = ''
    out_df['color'] = ''

    # TODO the names column in this map should actually be user defined (experiment description file?)
    channel_meta_map = pd.DataFrame(LIGHT_SOURCE_MAP)
    led_states = pd.DataFrame(NEUROPHOTOMETRICS_LED_STATES).set_index('Condition')
    states = raw_df['LedState']

    for state in states.unique():
        ir, ic = np.where(led_states == state)
        # if not present, multiple LEDs are active
        if ic.size == 0:
            # find row
            ir = np.argmax(led_states['No LED ON'] > state) - 1
            # find active combo
            possible_led_combos = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
            for combo in possible_led_combos:  # drop enumerate
                if state == sum([led_states.iloc[ir, c] for c in combo]):
                    name = '+'.join([channel_meta_map['name'][c] for c in combo])
                    color = '+'.join([channel_meta_map['color'][c] for c in combo])
                    wavelength = np.nan
                    out_df.loc[states == state, ['name', 'color', 'wavelength']] = (
                        name,
                        color,
                        wavelength,
                    )
        else:
            for cn in ['name', 'color', 'wavelength']:
                out_df.loc[states == state, cn] = channel_meta_map.iloc[ic[0]][cn]

    return out_df


def from_raw_neurophotometrics(path: str | Path) -> nap.TsdFrame:
    warnings.warn(
        'loading a photometry from raw neurophotometrics output. The data will _not_ be synced and\
            is being split into channels by LedState (converted to LED wavelength in nm)'
    )
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == '.csv':
        # really raw as it comes out of the device
        # todo figure out the header
        raw_df = pd.read_csv(path)
    if path.suffix == '.pqt':
        # as it is stored
        raw_df = pd.read_parquet(path)
    else:
        raise NotImplementedError

    df = _read_raw_neurophotometrics_df(raw_df)

    read_config = dict(
        data_columns=raw_df.columns[4:],
        time_column='times',
        channel_column='name',
    )
    return from_dataframe(df, **read_config)
