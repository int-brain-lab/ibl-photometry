from typing import Optional, Dict, List
import numpy as np
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from iblphotometry.fpio import _infer_data_columns, validate_photometry_df, from_photometry_df

"""
Neurophotometrics FP3002 specific information.
The light source map refers to the available LEDs on the system.
The flags refers to the byte encoding of led states in the system.
"""

LIGHT_SOURCE_MAP = {
    'color': ['None', 'Violet', 'Blue', 'Green'],
    'wavelength': [0, 415, 470, 560],
    'name': ['None', 'Isosbestic', 'GCaMP', 'RCaMP'],
}

LED_STATES = {
    'Condition': {
        0: 'No additional signal',
        1: 'Output 1 signal HIGH',
        2: 'Output 0 signal HIGH',
        3: 'Stimulation ON',
        4: 'GPIO Line 2 HIGH',
        5: 'GPIO Line 3 HIGH',
        6: 'Input 1 HIGH',
        7: 'Input 0 HIGH',
        8: 'Output 0 signal HIGH + Stimulation',
        9: 'Output 0 signal HIGH + Input 0 signal HIGH',
        10: 'Input 0 signal HIGH + Stimulation',
        11: 'Output 0 HIGH + Input 0 HIGH + Stimulation',
    },
    'No LED ON': {
        0: 0,
        1: 8,
        2: 16,
        3: 32,
        4: 64,
        5: 128,
        6: 256,
        7: 512,
        8: 48,
        9: 528,
        10: 544,
        11: 560,
    },
    'L415': {
        0: 1,
        1: 9,
        2: 17,
        3: 33,
        4: 65,
        5: 129,
        6: 257,
        7: 513,
        8: 49,
        9: 529,
        10: 545,
        11: 561,
    },
    'L470': {
        0: 2,
        1: 10,
        2: 18,
        3: 34,
        4: 66,
        5: 130,
        6: 258,
        7: 514,
        8: 50,
        9: 530,
        10: 546,
        11: 562,
    },
    'L560': {
        0: 4,
        1: 12,
        2: 20,
        3: 36,
        4: 68,
        5: 132,
        6: 260,
        7: 516,
        8: 52,
        9: 532,
        10: 548,
        11: 564,
    },
}


neurophotometrics_schemas = {
    'version_1': {
        'FrameCounter': pa.Column(pa.Int64),
        'Timestamp': pa.Column(pa.Float64),
        'Flags': pa.Column(pa.Int16, coerce=True),
    },
    'version_2': {
        'FrameCounter': pa.Column(pa.Int64),
        'Timestamp': pa.Column(pa.Float64),
        'LedState': pa.Column(pa.Int16, coerce=True),
    },
    'version_5': {
        'FrameCounter': pa.Column(pa.Int64),
        'SystemTimestamp': pa.Column(pa.Float64),
        'LedState': pa.Column(pa.Int16, coerce=True),
        'ComputerTimestamp': pa.Column(pa.Float64),
    },
}


def infer_neurophotometrics_version_from_data(df: pd.DataFrame) -> str:
    """
    Infer the neurophotometrics file version from DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Version string (e.g., 'version_1', 'version_2', etc.).

    Raises:
        ValueError: If no matching version is found for the input data.
    """
    # parse the data column format
    data_columns = _infer_data_columns(df)

    for version, schema in neurophotometrics_schemas.items():
        schema_ = pa.DataFrameSchema(
            columns=dict(
                **schema,
                **{k: pa.Column(pa.Float64) for k in data_columns},
            )
        )
        try:
            schema_.validate(df)
            return version  # since they are mutually exclusive return the first hit
        except SchemaError:
            # all fine, try next
            ...
    # if all attemps fail:
    raise ValueError('no matching version found for input data')


def read_neurophotometrics_file(path: str | Path) -> pd.DataFrame:
    """
    Read a neurophotometrics file (.csv or .pqt) into a DataFrame.

    Args:
        path (str | Path): Path to the file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If file format is unsupported.
    """
    path = Path(path) if isinstance(path, str) else path
    match path.suffix:
        case '.csv':
            raw_df = pd.read_csv(path)
        case '.pqt':
            raw_df = pd.read_parquet(path)
        case _:
            raise ValueError('unsupported file format')
    return raw_df


def from_neurophotometrics_df_to_photometry_df(
    raw_df: pd.DataFrame,
    version: Optional[str] = None,
    validate: bool = True,
    data_columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Convert a neurophotometrics DataFrame to a the ibl internal standardized photometry DataFrame.

    Args:
        raw_df (pd.DataFrame): Raw neurophotometrics DataFrame.
        version (str | None): Version string. If None, inferred automatically.
        validate (bool): Whether to validate the output DataFrame.
        data_columns (Optional[List[str]]): List of data columns. If None, inferred automatically.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        pd.DataFrame: Standardized photometry DataFrame.

    Raises:
        ValueError: If unknown version is provided.
    """
    if data_columns is None:
        data_columns = _infer_data_columns(raw_df)

    if version is None:
        version = infer_neurophotometrics_version_from_data(raw_df)

    # modify block - here all version specific adjustments will be made
    match version:
        case 'version_1':
            raw_df.rename(columns={'Flags': 'LedState'}, inplace=True)
            raw_df['valid'] = True
            raw_df['valid'] = raw_df['valid'].astype('bool')

        case 'version_2':
            raw_df['valid'] = True
            raw_df['valid'] = raw_df['valid'].astype('bool')

        case 'version_3':
            ...
        case 'version_4':
            ...
        case 'version_5':
            raw_df.rename(columns={'SystemTimestamp': 'Timestamp'}, inplace=True)
            raw_df['valid'] = True
            raw_df['valid'] = raw_df['valid'].astype('bool')
        case _:
            raise ValueError(f'unknown version {version}')  # should be impossible though

    photometry_df = raw_df.filter(items=data_columns, axis=1).sort_index(axis=1)
    photometry_df['times'] = raw_df['Timestamp']  # covered by validation now
    photometry_df['wavelength'] = np.nan
    photometry_df['name'] = ''
    photometry_df['color'] = ''
    photometry_df['valid'] = raw_df['valid']

    # TODO the names column in channel_meta_map should actually be user defined (experiment description file?)
    channel_meta_map = pd.DataFrame(LIGHT_SOURCE_MAP)
    led_states = pd.DataFrame(LED_STATES).set_index('Condition')

    # much cleaner code - should be the same functionality though
    # # decode led_state
    # possible_led_combos = [(0,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    # states_map = np.concatenate([np.sum(led_states.values[:,combo], axis=1)[:,np.newaxis] for combo in possible_led_combos],axis=1)

    # #
    # def decode_led_state(led_state: int):
    #     i, j = np.where(states_map == led_state)
    #     condition = led_states.index[i[0]]
    #     combo = possible_led_combos[j[0]]
    #     name = '+'.join([channel_meta_map['name'][c] for c in combo])
    #     color = '+'.join([channel_meta_map['color'][c] for c in combo])
    #     return name, color, condition

    # for state, group in raw_df.groupby('LedState'):
    #     raw_df.loc[group.index, 'color'] = decode_led_state(state)[1]

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
                    photometry_df.loc[states == state, ['name', 'color', 'wavelength']] = (
                        name,
                        color,
                        wavelength,
                    )
        else:
            for cn in ['name', 'color', 'wavelength']:
                photometry_df.loc[states == state, cn] = channel_meta_map.iloc[ic[0]][cn]

    # drop first frame
    if drop_first:
        photometry_df = photometry_df.iloc[1:].reset_index()

    if validate:
        photometry_df = validate_photometry_df(photometry_df, data_columns=data_columns)
    return photometry_df


def from_neurophotometrics_file_to_photometry_df(
    path: str | Path,
    version: Optional[str] = None,
    validate: bool = True,
    data_columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to read a neurophotometrics file and convert to photometry DataFrame.

    Args:
        path (str | Path): Path to the file.
        version (str | None): Version string. If None, inferred automatically.
        validate (bool): Whether to validate the output DataFrame.
        data_columns (Optional[List[str]]): List of data columns. If None, inferred automatically.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        pd.DataFrame: Standardized photometry DataFrame.
    """

    raw_df = read_neurophotometrics_file(path)
    photometry_df = from_neurophotometrics_df_to_photometry_df(
        raw_df,
        version=version,
        validate=validate,
        data_columns=data_columns,
        drop_first=drop_first,
    )
    return photometry_df


def from_neurophotometrics_file(
    path: str | Path,
    drop_first: bool = True,
    validate: bool = True,
    version: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Read a neurophotometrics file and split into channel DataFrames.

    Args:
        path (str | Path): Path to the file.
        drop_first (bool): Whether to drop the first frame.
        validate (bool): Whether to validate the DataFrame.
        version (str | None): Version string. If None, inferred automatically.

    Returns:
        dict: Dictionary of DataFrames per channel.
    """
    photometry_df = from_neurophotometrics_file_to_photometry_df(
        path,
        drop_first=drop_first,
        validate=validate,
        version=version,
    )
    return from_photometry_df(photometry_df)


"""
########  ####  ######   #### ########    ###    ##          #### ##    ## ########  ##     ## ########  ######
##     ##  ##  ##    ##   ##     ##      ## ##   ##           ##  ###   ## ##     ## ##     ##    ##    ##    ##
##     ##  ##  ##         ##     ##     ##   ##  ##           ##  ####  ## ##     ## ##     ##    ##    ##
##     ##  ##  ##   ####  ##     ##    ##     ## ##           ##  ## ## ## ########  ##     ##    ##     ######
##     ##  ##  ##    ##   ##     ##    ######### ##           ##  ##  #### ##        ##     ##    ##          ##
##     ##  ##  ##    ##   ##     ##    ##     ## ##           ##  ##   ### ##        ##     ##    ##    ##    ##
########  ####  ######   ####    ##    ##     ## ########    #### ##    ## ##         #######     ##     ######
"""
"""
##     ##    ###    ##       #### ########     ###    ######## ####  #######  ##    ##
##     ##   ## ##   ##        ##  ##     ##   ## ##      ##     ##  ##     ## ###   ##
##     ##  ##   ##  ##        ##  ##     ##  ##   ##     ##     ##  ##     ## ####  ##
##     ## ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ## ## ##
 ##   ##  ######### ##        ##  ##     ## #########    ##     ##  ##     ## ##  ####
  ## ##   ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ##   ###
   ###    ##     ## ######## #### ########  ##     ##    ##    ####  #######  ##    ##
"""
neurophotometrics_digital_inputs_schemas = {
    'version_1': {
        'Timestamp': pa.Column(pa.Float64),
        'Value': pa.Column(pa.Bool, coerce=True),
    },
    'version_2': {
        'Timestamp': pa.Column(pa.Float64),
        'Value.Seconds': pa.Column(pa.Float64),
    },
    'version_5': {
        'ChannelName': pa.Column(pa.String),
        'Channel': pa.Column(pa.Int8),
        'AlwaysTrue': pa.Column(pa.Bool),
        'SystemTimestamp': pa.Column(pa.Float64),
        'ComputerTimestamp': pa.Column(pa.Float64),
    },
}

digital_input_schema = {
    'times': pa.Column(pa.Float64),
    'channel_name': pa.Column(str, coerce=True),
    'channel': pa.Column(pa.Int8, coerce=True),
    'polarity': pa.Column(pa.Int8),
}


def infer_neurophotometrics_version_from_digital_inputs(df: pd.DataFrame) -> str:
    """
    Infer the neurophotometrics digital inputs file version from DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Version string.

    Raises:
        ValueError: If no matching version is found.
    """
    for version, schema in neurophotometrics_digital_inputs_schemas.items():
        schema_ = pa.DataFrameSchema(columns=dict(**schema))
        try:
            schema_.validate(df)
            return version  # since they are mutually exclusive return the first hit
        except SchemaError:
            # all fine, try next
            ...
    # if all attemps fail:
    raise ValueError('no matching version found')


"""
########  ########    ###    ########  ######## ########
##     ## ##         ## ##   ##     ## ##       ##     ##
##     ## ##        ##   ##  ##     ## ##       ##     ##
########  ######   ##     ## ##     ## ######   ########
##   ##   ##       ######### ##     ## ##       ##   ##
##    ##  ##       ##     ## ##     ## ##       ##    ##
##     ## ######## ##     ## ########  ######## ##     ##
"""


def read_digital_inputs_file(
    path: str | Path,
    version: Optional[str] = None,
    validate: bool = True,
    channel: Optional[int] = None,
    timestamps_colname: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(path) if isinstance(path, str) else path
    match path.suffix:
        case '.csv':
            df = pd.read_csv(path)
        case '.pqt':
            df = pd.read_parquet(path)
        case _:
            raise ValueError('unsupported file format')

    if validate:
        df = validate_digital_inputs_df(
            df,
            version=version,
            channel=channel,
            timestamps_colname=timestamps_colname,
        )
    return df


def validate_digital_inputs_df(
    df: pd.DataFrame,
    version: Optional[str] = None,
    validate: bool = True,
    channel: Optional[int] = None,
    timestamps_colname: Optional[str] = None,
) -> pd.DataFrame:
    if version is None:
        version = infer_neurophotometrics_version_from_digital_inputs(df)

    # modify block - here all version specific adjustments will be made
    match version:
        case 'version_1':
            assert channel is not None, 'attempting to load an old file version without explicitly knowing the channel'
            df['channel'] = channel
            df['channel_name'] = f'channel_{channel}'
            df['channel'] = df['channel'].astype('int64')
            df = df.rename(columns={'Timestamp': 'times', 'Value': 'polarity'})
            df['polarity'] = df['polarity'].replace({True: 1, False: -1}).astype('int8')

        case 'version_2':
            assert channel is not None, 'attempting to load an old file version without explicitly knowing the channel'
            assert timestamps_colname is not None, 'for version 2, column name for timestamps need to be provided'
            assert timestamps_colname in {'Value.Seconds', 'Timestamp'}, (
                f'timestamps_colname needs to be either Value.Seconds or Timestamp, but is {timestamps_colname}'
            )
            df['channel'] = channel
            df['channel_name'] = f'channel_{channel}'
            df['times'] = df[timestamps_colname]
            df = df.rename(columns={'Value.Value': 'polarity'})
            df['polarity'] = df['polarity'].replace(
                {
                    True: 1,
                    False: -1,
                }
            )  # FIXME causes downcasting warning, see https://github.com/pandas-dev/pandas/issues/57734
            df = df.astype({'polarity': 'int8', 'channel': 'int64'})
            df = df.drop(['Timestamp', 'Value.Seconds'], axis=1)

        case 'version_3':
            ...
        case 'version_4':
            ...
        case 'version_5':
            df = df.rename(
                columns={
                    'ChannelName': 'channel_name',
                    'SystemTimestamp': 'times',
                    'Channel': 'channel',
                }
            )
            df = df.drop(['AlwaysTrue', 'ComputerTimestamp'], axis=1)
            df['polarity'] = 1
            df = df.astype({'polarity': 'int8'})
        case _:
            raise ValueError(f'unknown version {version}')  # should be impossible though

    df = pa.DataFrameSchema(columns=digital_input_schema).validate(df)
    return df
