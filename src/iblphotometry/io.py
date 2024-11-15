# %%
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
import warnings
from iblphotometry.neurophotometrics_definitions import NEUROPHOTOMETRICS_LED_STATES


# def from_csv(): ...


def from_dataframe(
    raw_df: pd.DataFrame,
    data_columns: list[str] = None,
    time_column: str = None,
    channel_column: str = 'name',
    channel_names: list[str] = None,
):
    # from a raw dataframe as it is stored in ONE (signal.pqt)
    # data_columns is a list of str that specifies the names of the column that hold the actual data, like 'RegionXX'
    # channel_column is the column that specifies the temporally multiplexed acquisition channels
    if data_columns is None:
        # infer if not explicitly provided: defaults to everything that starts with 'Region'
        data_columns = [col for col in raw_df.columns if col.startswith('Region')]

    if time_column is None:
        # infer name of time column if not provided
        (time_column,) = [col for col in raw_df.columns if 'time' in col.lower()]

    if channel_names is None:
        # infer channel names if they are not explicitly provided
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
        # TODO think about the renaming feature
        # relevant for the user is not 'RegionXX' but rather brain_region or similar
        # if rename:
        #     rename_map = self.get_mapping(eid, key='ROI', value='brain_region')
        #     raw_photometry_df = raw_photometry_df.rename(rename_map)
        df = raw_df.groupby(channel_column).get_group(channel)
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


def from_raw_neurophotometrics(path: str | Path):
    warnings.warn(
        'loading a photometry from raw neurophotometrics output. The data will _not_ be synced and\
            is being split into channels by LedState (converted to LED wavelength in nm)'
    )
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == '.csv':
        # todo figure out the header
        raw_df = pd.read_csv(path)
    if path.suffix == '.pqt':
        raw_df = pd.read_parquet(path)
    else:
        raise NotImplementedError

    # led_states = raw_df['LedState'].unique()
    # converting less meaninful ledstate code to nm value of the active LED
    led_states_df = pd.DataFrame(NEUROPHOTOMETRICS_LED_STATES).set_index('Condition')
    # FIXME
    # this currently only takes care of the condition 'not additional signal'
    # find out when this condition is true and when not
    # (it is true for the example file present)
    condition = 'No additional signal'
    led_states = led_states_df.loc[condition]
    led_state_to_nm = dict(zip(led_states.values, led_states.index))
    raw_df['Led_nm'] = [
        led_state_to_nm[led_state] for led_state in raw_df['LedState'].values
    ]

    read_config = dict(
        channel_column='Led_nm',
        time_column='Timestamp',
    )

    return from_dataframe(raw_df, **read_config)
