import numpy as np
import pandas as pd
from pathlib import Path
import pandera.pandas as pa
from pandera.errors import SchemaError
from one.api import ONE

from iblphotometry.neurophotometrics import (
    LIGHT_SOURCE_MAP,
    LED_STATES,
)


"""
##    ## ########  ##     ##    ######## #### ##       ########  ######
###   ## ##     ## ##     ##    ##        ##  ##       ##       ##    ##
####  ## ##     ## ##     ##    ##        ##  ##       ##       ##
## ## ## ########  #########    ######    ##  ##       ######    ######
##  #### ##        ##     ##    ##        ##  ##       ##             ##
##   ### ##        ##     ##    ##        ##  ##       ##       ##    ##
##    ## ##        ##     ##    ##       #### ######## ########  ######
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

photometry_df_schema = {
    'times': pa.Column(pa.Float64),
    'valid': pa.Column(pa.Bool),
    'wavelength': pa.Column(pa.Float64, nullable=True),
    'name': pa.Column(pa.String),  # this should rather be "channel_name" or "channel"
    'color': pa.Column(pa.String),
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


def _infer_data_columns(df: pd.DataFrame) -> list[str]:
    # small helper, returns the data columns from a photometry dataframe
    if any([col.startswith('Region') for col in df.columns]):
        data_columns = [col for col in df.columns if col.startswith('Region')]
    else:
        data_columns = [col for col in df.columns if col.startswith('R') or col.startswith('G')]
    return data_columns


def validate_photometry_df(
    photometry_df: pd.DataFrame,
    data_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Validate the photometry DataFrame against the schema.

    Args:
        photometry_df (pd.DataFrame): Input DataFrame.
        data_columns (list[str] | None): List of data columns to validate. If None, inferred automatically.

    Returns:
        pd.DataFrame: Validated DataFrame.

    Raises:
        SchemaError: If validation fails.
    """
    data_columns = _infer_data_columns(photometry_df) if data_columns is None else data_columns
    schema = pa.DataFrameSchema(
        columns=dict(
            **photometry_df_schema,
            **{k: pa.Column(pa.Float64) for k in data_columns},
        )
    )
    return schema.validate(photometry_df)


"""
########  ########    ###    ########  ######## ########   ######
##     ## ##         ## ##   ##     ## ##       ##     ## ##    ##
##     ## ##        ##   ##  ##     ## ##       ##     ## ##
########  ######   ##     ## ##     ## ######   ########   ######
##   ##   ##       ######### ##     ## ##       ##   ##         ##
##    ##  ##       ##     ## ##     ## ##       ##    ##  ##    ##
##     ## ######## ##     ## ########  ######## ##     ##  ######
"""


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
    version: str | None = None,
    validate: bool = True,
    data_columns: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Convert a neurophotometrics DataFrame to a the ibl internal standardized photometry DataFrame.

    Args:
        raw_df (pd.DataFrame): Raw neurophotometrics DataFrame.
        version (str | None): Version string. If None, inferred automatically.
        validate (bool): Whether to validate the output DataFrame.
        data_columns (list[str] | None): List of data columns. If None, inferred automatically.
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
    version: str | None = None,
    validate: bool = True,
    data_columns: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to read a neurophotometrics file and convert to photometry DataFrame.

    Args:
        path (str | Path): Path to the file.
        version (str | None): Version string. If None, inferred automatically.
        validate (bool): Whether to validate the output DataFrame.
        data_columns (list[str] | None): List of data columns. If None, inferred automatically.
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


def from_photometry_df(
    photometry_df: pd.DataFrame,
    data_columns: list[str] | None = None,
    channel_names: list[str] | None = None,
    rename: dict | None = None,  # the dict to rename the data_columns -> Region?G | G? -> brain_region
    validate: bool = True,
    drop_first: bool = True,
) -> dict[pd.DataFrame]:
    """
    Split a photometry DataFrame into separate DataFrames per channel.

    Args:
        photometry_df (pd.DataFrame): Input photometry DataFrame.
        data_columns (list[str] | None): List of data columns. If None, inferred automatically.
        channel_names (list[str] | None): List of channel names. If None, inferred automatically.
        rename (dict | None): Mapping to rename data columns.
        validate (bool): Whether to validate the DataFrame.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        dict[pd.DataFrame]: Dictionary of DataFrames per channel.
    """
    if validate:
        photometry_df = validate_photometry_df(photometry_df)

    data_columns = _infer_data_columns(photometry_df) if data_columns is None else data_columns

    # drop first?
    if drop_first:
        photometry_df = photometry_df.iloc[1:]

    # infer channel names if they are not explicitly provided
    if channel_names is None:
        channel_names = photometry_df['name'].unique()

    # drop empty acquisition channels
    to_drop = ['None', '']
    channel_names = [ch for ch in channel_names if ch not in to_drop]

    signal_dfs = {}
    for channel in channel_names:
        # get the data for the band
        df = photometry_df.groupby('name').get_group(channel)
        # if rename dict is passed, rename Region0X to the corresponding brain region
        if rename is not None:
            df = df.rename(columns=rename)
            data_columns = rename.values()
        signal_dfs[channel] = df.set_index('times')[data_columns]

    return signal_dfs


def from_photometry_pqt(
    photometry_pqt_path: str | Path,
    locations_pqt_path: str | Path | None = None,
    drop_first=True,
) -> dict[pd.DataFrame]:
    """
    Load photometry and location data from parquet files and split by channel.

    Args:
        photometry_pqt_path (str | Path): Path to photometry parquet file.
        locations_pqt_path (str | Path | None): Path to locations parquet file.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        dict[pd.DataFrame]: Dictionary of DataFrames per channel.
    """
    photometry_df = pd.read_parquet(photometry_pqt_path)

    if locations_pqt_path is not None:
        locations_df = pd.read_parquet(locations_pqt_path)
        data_columns = (list(locations_df.index),)
        rename = locations_df['brain_region'].to_dict()
    else:
        # warnings.warn('loading a photometry.signal.pqt file without its corresponding photometryROI.locations.pqt')
        data_columns = None
        rename = None

    return from_photometry_df(
        photometry_df,
        data_columns=data_columns,
        rename=rename,
        drop_first=drop_first,
    )


def from_neurophotometrics_file(
    path: str | Path,
    drop_first: bool = True,
    validate: bool = True,
    version: str | None = None,
) -> dict:
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


def from_eid(eid: str, one) -> list[dict]:
    """
    Load photometry data for a session ID (eid) using ONE.

    Args:
        eid (str): Session ID.
        one: ONE API instance.

    Returns:
        list[dict]: List of channel DataFrames.
    """
    one.load_dataset(eid, 'alf/photometry/photometry.signal.pqt', download_only=True)
    one.load_dataset(eid, 'alf/photometry/photometryROI.locations.pqt', download_only=True)
    session_path = one.eid2path(eid)
    return from_session_path(session_path)


def from_session_path(session_path: str | Path, drop_first: bool = True) -> list[dict]:
    """
    Load photometry data from a locally present session path.

    Args:
        session_path (str | Path): Path to session folder.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        list[dict]: List of channel DataFrames.
    """
    session_path = Path(session_path) if isinstance(session_path, str) else session_path
    assert session_path.exists()
    return from_photometry_pqt(
        session_path / 'alf/photometry/photometry.signal.pqt',
        session_path / 'alf/photometry/photometryROI.locations.pqt',
        drop_first=drop_first,
    )


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
    version: str | None = None,
    validate: bool = True,
    channel: int | None = None,
    timestamps_colname: str | None = None,
) -> pd.DataFrame:
    """
    Read and standardize a digital inputs file.

    Args:
        path (str | Path): Path to the file.
        version (str | None): Version string. If None, inferred automatically.
        validate (bool): Whether to validate the output DataFrame.
        channel (int | None): Channel number (required for old versions).
        timestamps_colname (str | None): Column name for timestamps (required for version 2).

    Returns:
        pd.DataFrame: Standardized digital inputs DataFrame.

    Raises:
        ValueError: If file format or version is unsupported.
        AssertionError: If required arguments are missing for old versions.
    """
    path = Path(path) if isinstance(path, str) else path
    match path.suffix:
        case '.csv':
            df = pd.read_csv(path)
        case '.pqt':
            df = pd.read_parquet(path)
        case _:
            raise ValueError('unsupported file format')

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
            assert timestamps_colname == 'Value.Seconds' or timestamps_colname == 'Timestamp', (
                f'timestamps_colname needs to be either Value.Seconds or Timestamp, but is {timestamps_colname}'
            )
            df['channel'] = channel
            df['channel_name'] = f'channel_{channel}'
            df['times'] = df[timestamps_colname]
            df = df.rename(columns={'Value.Value': 'polarity'})
            df['polarity'] = df['polarity'].replace(
                {True: 1, False: -1}
            )  # FIXME causes downcasting warning, see https://github.com/pandas-dev/pandas/issues/57734
            df = df.astype({'polarity': 'int8', 'channel': 'int64'})
            df = df.drop(['Timestamp', 'Value.Seconds'], axis=1)

        case 'version_3':
            ...
        case 'version_4':
            ...
        case 'version_5':
            df = df.rename(columns={'ChannelName': 'channel_name', 'SystemTimestamp': 'times', 'Channel': 'channel'})
            df = df.drop(['AlwaysTrue', 'ComputerTimestamp'], axis=1)
            df['polarity'] = 1
            df = df.astype({'polarity': 'int8'})
        case _:
            raise ValueError(f'unknown version {version}')  # should be impossible though

    if validate:
        df = pa.DataFrameSchema(columns=digital_input_schema).validate(df)
    return df
