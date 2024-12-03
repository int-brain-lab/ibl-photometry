# %%
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pandera

from iblphotometry.neurophotometrics import (
    LIGHT_SOURCE_MAP,
    LED_STATES,
)


def from_array(
    times: np.array, data: np.array, channel_names: list[str] = None
) -> pd.DataFrame:
    return pd.DataFrame(data, index=times, columns=channel_names)


def from_dataframe(
    raw_df: pd.DataFrame,
    data_columns: list[str] = None,
    time_column: str = None,
    channel_column: str = 'name',
    channel_names: list[str] = None,
    rename: dict = None,
) -> dict:
    """reads in a pandas.DataFrame and converts it into nap.TsdDataframes. Performs the time demultiplexing operation.


    Args:
        raw_df (pd.DataFrame): the dataframe, as stored in the photometry.signal.pqt
        data_columns (list[str], optional): The names of the columns in the dataframe that contain the signals of different fibers. By default, they are named RegionXX. If None is provided, All columns that start with `Region` are treated as data columns. Defaults to None.
        time_column (str, optional): The name of the column that contains the timestamps. If None is provided, it is assumed that `time` is in the name. Defaults to None.
        channel_column (str, optional): The name of the column that contains. Defaults to 'name'.
        channel_names (list[str], optional): The names of the acquisition channel / frequency bands that are acquired. Defaults to None.
        rename (dict, optional): a renaming map that maps the names of the columns to brain areas. Example: {'RegionXX':'DMS'}. Defaults to None.

    Returns:
        dict: A dict with the keys being the names of the acquisition channels, the values being nap.TsdFrames with the columns containing the data of the different fibers
    """
    # from a raw dataframe as it is stored in ONE (signal.pqt)
    # data_columns is a list of str that specifies the names of the column that hold the actual data, like 'RegionXX'
    # channel_column is the column that specifies the temporally multiplexed acquisition channels

    # infer if not explicitly provided: defaults to everything that starts with 'Region'
    if data_columns is None:
        data_columns = [col for col in raw_df.columns if col.startswith('Region')]

    # infer name of time column if not provided
    if time_column is None:
        time_column = [col for col in raw_df.columns if 'time' in col.lower()]
        assert len(time_column) == 1
        time_column = time_column[0]

    # infer channel names if they are not explicitly provided
    if channel_names is None:
        channel_names = raw_df[channel_column].unique()

    # drop empty acquisition channels
    to_drop = ['None', '']
    channel_names = [ch for ch in channel_names if ch not in to_drop]

    raw_dfs = {}
    for channel in channel_names:
        # get the data for the band
        df = raw_df.groupby(channel_column).get_group(channel)
        # if rename dict is passed, rename Region0X to the corresponding brain region
        if rename is not None:
            df = df.rename(columns=rename)
            data_columns = rename.values()
        raw_dfs[channel] = df.set_index(time_column)[data_columns]

    return raw_dfs


def from_pqt(
    signal_pqt_path: str | Path,
    locations_pqt_path: str | Path = None,
):
    """reads in a photometry.signal.pqt files as they are registered in alyx.

    Args:
        signal_pqt_path (str | Path): _description_
        locations_pqt_path (str | Path, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    raw_df = pd.read_parquet(signal_pqt_path)
    if locations_pqt_path is not None:
        locations_df = pd.read_parquet(locations_pqt_path)
        data_columns = (list(locations_df.index),)
        rename = locations_df['brain_region'].to_dict()
    else:
        warnings.warn(
            'loading a photometry.signal.pqt file without its corresponding photometryROI.locations.pqt'
        )
        data_columns = None
        rename = None

    read_config = dict(
        data_columns=data_columns,
        time_column='times',
        channel_column='name',
        rename=rename,
    )

    return from_dataframe(raw_df, **read_config)


def _read_raw_neurophotometrics_df(raw_df: pd.DataFrame, rois=None) -> pd.DataFrame:
    """reads in parses the output of the neurophotometrics FP3002

    Args:
        raw_df (pd.DataFrame): _description_
        rois (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: a dataframe in the same format as stored in alyx as pqt.
    """
    if rois is None:
        rois = [col for col in raw_df.columns if col.startswith('G')]

    out_df = raw_df.filter(items=rois, axis=1).sort_index(axis=1)
    timestamp_name = (
        'SystemTimestamp' if 'SystemTimestamp' in raw_df.columns else 'Timestamp'
    )
    out_df['times'] = raw_df[timestamp_name]
    out_df['wavelength'] = np.nan
    out_df['name'] = ''
    out_df['color'] = ''

    # TODO the names column in channel_meta_map should actually be user defined (experiment description file?)
    channel_meta_map = pd.DataFrame(LIGHT_SOURCE_MAP)
    led_states = pd.DataFrame(LED_STATES).set_index('Condition')
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


def from_raw_neurophotometrics(
    path: str | Path,
    drop_first=True,
    validate=True,
) -> dict:
    """reads a raw neurophotometrics file (in .csv or .pqt format) as they are written by the neurophotometrics software

    Args:
        path (str | Path): path to either the .csv file as written by the neurophotometrics bonsai workflow, or a path to a .pqt file as stored in alyx
        drop_first (bool, optional): The first frame is all LEDs on. If true, this frame is dropped. Defaults to True.
        validate (bool, optional): if true, enforces pydantic validation of the datatypes. Defaults to TRue

    Raises:
        NotImplementedError: _description_

    Returns:
        nap.TsdFrame: _description_ # FIXME
    """
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
    elif path.suffix == '.pqt':
        # as it is stored
        raw_df = pd.read_parquet(path)
    else:
        raise NotImplementedError

    if validate:
        raw_df = _validate_dataframe(raw_df)

    df = _read_raw_neurophotometrics_df(raw_df)

    # drop first frame
    if drop_first:
        df = df.iloc[1:]

    data_columns = [col for col in df.columns if col.startswith('G')]
    read_config = dict(
        data_columns=data_columns,
        time_column='times',
        channel_column='name',
    )
    return from_dataframe(df, **read_config)


def _validate_dataframe(
    df: pd.DataFrame,
    data_columns=None,
) -> pd.DataFrame:
    if data_columns is None:
        data_columns = [col for col in df.columns if col.startswith('G')]

    schema_raw_data = pandera.DataFrameSchema(
        columns=dict(
            FrameCounter=pandera.Column(pandera.Int64),
            SystemTimestamp=pandera.Column(pandera.Float64),
            LedState=pandera.Column(pandera.Int16, coerce=True),
            ComputerTimestamp=pandera.Column(pandera.Float64),
            **{k: pandera.Column(pandera.Float64) for k in data_columns},
        )
    )

    return schema_raw_data.validate(df)

def _validate_neurophotometrics_digital_inputs(df: pd.DataFrame) -> pd.DataFrame:
    schema_digital_inputs = pandera.DataFrameSchema(
        columns=dict(
            ChannelName=pandera.Column(str, coerce=True),
            Channel=pandera.Column(pandera.Int8, coerce=True),
            AlwaysTrue=pandera.Column(bool, coerce=True),
            SystemTimestamp=pandera.Column(pandera.Float64),
            ComputerTimestamp=pandera.Column(pandera.Float64),
        )
    )
    return schema_digital_inputs.validate(df)

