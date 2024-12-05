import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pandera
from typing import Optional

from iblphotometry.neurophotometrics import (
    LIGHT_SOURCE_MAP,
    LED_STATES,
)


def from_raw_neurophotometrics_file_to_raw_df(
    path: str | Path,
    validate=True,
) -> pd.DataFrame:
    path = Path(path) if isinstance(path, str) else path
    match path.suffix:
        case '.csv':
            raw_df = pd.read_csv(path)
        case '.pqt':
            raw_df = pd.read_parquet(path)

    if validate:
        raw_df = _validate_neurophotometrics_df(raw_df)

    return raw_df


def from_raw_neurophotometrics_df_to_ibl_df(
    raw_df: pd.DataFrame, rois=None, drop_first=True
) -> pd.DataFrame:
    if rois is None:
        rois = infer_data_columns(raw_df)

    ibl_df = raw_df.filter(items=rois, axis=1).sort_index(axis=1)
    timestamp_name = (
        'SystemTimestamp' if 'SystemTimestamp' in raw_df.columns else 'Timestamp'
    )
    ibl_df['times'] = raw_df[timestamp_name]
    ibl_df['wavelength'] = np.nan
    ibl_df['name'] = ''
    ibl_df['color'] = ''

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
                    ibl_df.loc[states == state, ['name', 'color', 'wavelength']] = (
                        name,
                        color,
                        wavelength,
                    )
        else:
            for cn in ['name', 'color', 'wavelength']:
                ibl_df.loc[states == state, cn] = channel_meta_map.iloc[ic[0]][cn]

    # drop first frame
    if drop_first:
        ibl_df = ibl_df.iloc[1:].reset_index()

    return ibl_df


def from_raw_neurophotometrics_file_to_ibl_df(
    path: str | Path,
    drop_first=True,
    validate=True,
) -> pd.DataFrame:
    raw_df = from_raw_neurophotometrics_file_to_raw_df(path, validate=validate)
    ibl_df = from_raw_neurophotometrics_df_to_ibl_df(raw_df, drop_first=drop_first)

    return ibl_df


def from_ibl_pqt_to_ibl_df(path: str | Path, validate=False):
    if validate is True:
        # TODO
        raise NotImplementedError
    return pd.read_parquet(path)


def from_ibl_dataframe(
    ibl_df: pd.DataFrame,
    data_columns: list[str] | None = None,
    time_column: str | None = None,
    channel_column: str = 'name',
    channel_names: list[str] | None = None,
    rename: dict | None = None,
    validate: bool = True,
) -> dict:
    """main function to convert to analysis ready format


    Args:
        ibl_df (pd.DataFrame): the dataframe, as stored in the photometry.signal.pqt
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

    data_columns = infer_data_columns(ibl_df) if data_columns is None else data_columns

    # infer name of time column if not provided
    if time_column is None:
        time_columns = [col for col in ibl_df.columns if 'time' in col.lower()]
        assert len(time_columns) == 1
        time_column = time_columns[0]

    # infer channel names if they are not explicitly provided
    if channel_names is None:
        channel_names = ibl_df[channel_column].unique()

    # drop empty acquisition channels
    if validate:
        ibl_df = validate_ibl_dataframe(ibl_df)

    dfs = {}
    for channel in channel_names:
        # get the data for the band
        df = ibl_df.groupby(channel_column).get_group(channel)
        # if rename dict is passed, rename Region0X to the corresponding brain region
        if rename is not None:
            df = df.rename(columns=rename)
            data_columns = rename.values()
        dfs[channel] = df.set_index(time_column)[data_columns]

    return dfs


def from_ibl_pqt(
    signal_pqt_path: str | Path,
    locations_pqt_path: Optional[str | Path] = None,
):
    # read from a single pqt
    # if both are provided, do both

    ibl_df = pd.read_parquet(signal_pqt_path)
    if locations_pqt_path is not None:
        locations_df = pd.read_parquet(locations_pqt_path)
        return from_ibl_dataframes(ibl_df, locations_df)
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

    return from_ibl_dataframe(ibl_df, **read_config)


def from_ibl_dataframes(ibl_df: pd.DataFrame, locations_df: pd.DataFrame):
    # if locations are present
    data_columns = (list(locations_df.index),)
    rename = locations_df['brain_region'].to_dict()

    read_config = dict(
        data_columns=data_columns,
        time_column='times',
        channel_column='name',
        rename=rename,
    )

    return from_ibl_dataframe(ibl_df, **read_config)


def from_raw_neurophotometrics_file(
    path: str | Path,
    drop_first=True,
    validate=True,
) -> dict:
    # this one bypasses everything
    ibl_df = from_raw_neurophotometrics_file_to_ibl_df(
        path, drop_first=drop_first, validate=validate
    )
    # data_columns = infer_data_columns(ibl_df) if data_columns is None else data_columns
    read_config = dict(
        # data_columns=data_columns,
        time_column='times',
        channel_column='name',
    )
    return from_ibl_dataframe(ibl_df, **read_config)


"""
##     ##    ###    ##       #### ########     ###    ######## ####  #######  ##    ##
##     ##   ## ##   ##        ##  ##     ##   ## ##      ##     ##  ##     ## ###   ##
##     ##  ##   ##  ##        ##  ##     ##  ##   ##     ##     ##  ##     ## ####  ##
##     ## ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ## ## ##
 ##   ##  ######### ##        ##  ##     ## #########    ##     ##  ##     ## ##  ####
  ## ##   ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ##   ###
   ###    ##     ## ######## #### ########  ##     ##    ##    ####  #######  ##    ##
"""


def validate_ibl_dataframe(ibl_df: pd.DataFrame) -> pd.DataFrame:
    # for now, check if number of frames are equal and drop the longer one
    # to be expanded into a full panderas check

    # 1) drop first frame if invalid
    first_frame_name = ibl_df.iloc[0]['name']
    if '+' in first_frame_name or first_frame_name == '':
        ibl_df = ibl_df.drop(index=0)

    # 2) if unequal number of frames per acquisition channel, drop excess frames
    frame_counts = ibl_df.groupby('name')['times'].count()
    if not np.all(frame_counts.values == frame_counts.values[0]):
        # find shortest
        dfs = []
        min_frames = frame_counts.iloc[np.argmin(frame_counts)]
        for name, group in ibl_df.groupby('name'):
            dfs.append(group.iloc[:min_frames])
            n_dropped = group.shape[0] - min_frames
            warnings.warn(f'dropping {n_dropped} frames for channel {name}')

        ibl_df = pd.concat(dfs).sort_index()

    # 3) panderas validation
    data_columns = infer_data_columns(ibl_df)
    schema_ibl_df = pandera.DataFrameSchema(
        columns=dict(
            times=pandera.Column(pandera.Float64),
            # valid=pandera.Column(pandera.Bool), # NOTE as of now, it seems like valid is an optional column found in alejandro but not in carolina
            wavelength=pandera.Column(pandera.Float64),
            name=pandera.Column(pandera.String),
            color=pandera.Column(pandera.String),
            **{k: pandera.Column(pandera.Float64) for k in data_columns},
        )
    )
    ibl_df = schema_ibl_df.validate(ibl_df)
    return ibl_df


def _validate_neurophotometrics_df(
    df: pd.DataFrame,
    data_columns=None,
) -> pd.DataFrame:
    data_columns = infer_data_columns(df) if data_columns is None else data_columns

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


def infer_data_columns(df: pd.DataFrame) -> list[str]:
    # this hacky parser currently deals with the inconsistency between carolinas and alejandros extraction
    # https://github.com/int-brain-lab/ibl-photometry/issues/35
    data_columns = [
        col for col in df.columns if col.startswith('Region') or col.startswith('G')
    ]
    return data_columns
