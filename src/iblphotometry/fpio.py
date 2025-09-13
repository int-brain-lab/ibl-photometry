import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pandera.pandas as pa
from pandera.errors import SchemaError
from typing import Optional, List
import os

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
    # parse the data column format
    data_columns = infer_data_columns(df)

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


def infer_data_columns(df: pd.DataFrame) -> List[str]:
    if any([col.startswith('Region') for col in df.columns]):
        data_columns = [col for col in df.columns if col.startswith('Region')]
    else:
        data_columns = [col for col in df.columns if col.startswith('R') or col.startswith('G')]
    return data_columns


def validate_photometry_df(photometry_df: pd.DataFrame, data_columns=None) -> pd.DataFrame:
    data_columns = infer_data_columns(photometry_df) if data_columns is None else data_columns
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
    data_columns: List[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    if data_columns is None:
        data_columns = infer_data_columns(raw_df)

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
    data_columns: List[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    # this is the convenience function

    raw_df = read_neurophotometrics_file(path)
    photometry_df = from_neurophotometrics_df_to_photometry_df(
        raw_df,
        version=version,
        validate=validate,
        data_columns=data_columns,
        drop_first=drop_first,
    )
    return photometry_df


# def from_ibl_pqt_to_photometry_df(
#     path: str | Path,
#     validate=True,  # could default to false because it should not be possible to write a non-validated photometry_df to disk
# ):
#     photometry_df = pd.read_parquet(path)
#     if validate is True:
#         photometry_df = validate_photometry_df(photometry_df)
#     return photometry_df


def from_photometry_df(
    photometry_df: pd.DataFrame,
    data_columns: list[str] | None = None,
    channel_names: list[str] | None = None,
    rename: dict | None = None,  # the dict to rename the data_columns -> Region?G | G? -> brain_region
    validate: bool = True,
) -> dict[pd.DataFrame]:
    # at this point this might be overkill ...
    if validate:
        photometry_df = validate_photometry_df(photometry_df)

    data_columns = infer_data_columns(photometry_df) if data_columns is None else data_columns

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
            df.rename(columns=rename, inplace=True)
            data_columns = rename.values()
        signal_dfs[channel] = df.set_index('times')[data_columns]

    return signal_dfs


def from_photometry_pqt(
    photometry_pqt_path: str | Path,
    locations_pqt_path: Optional[str | Path] = None,
) -> dict[pd.DataFrame]:
    """ """
    photometry_df = pd.read_parquet(photometry_pqt_path)

    if locations_pqt_path is not None:
        locations_df = pd.read_parquet(locations_pqt_path)
        data_columns = (list(locations_df.index),)
        rename = locations_df['brain_region'].to_dict()
    else:
        # warnings.warn('loading a photometry.signal.pqt file without its corresponding photometryROI.locations.pqt')
        data_columns = None
        rename = None

    return from_photometry_df(photometry_df, data_columns=data_columns, rename=rename)


def from_neurophotometrics_file(
    path: str | Path,
    drop_first: bool = True,
    validate: bool = True,
    version: str | None = None,
) -> dict:
    photometry_df = from_neurophotometrics_file_to_photometry_df(
        path,
        drop_first=drop_first,
        validate=validate,
        version=version,
    )
    return from_photometry_df(photometry_df)


def from_session_path(session_path: str | Path) -> List[dict]:
    # this should be the main user facing function
    # or the one to be recycled by the PhotometryLoader
    # load the data from the alf path
    #
    ...


def from_eid(eid: str, one) -> List[dict]:
    # ...
    ...


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


# def validate_digital_inputs(digital_inputs_df: pd.DataFrame) -> pd.DataFrame:
#     # DOCME
#     schema_digital_inputs = pa.DataFrameSchema(
#         columns=dict(
#             times=pa.Column(pa.Float64),
#             ChannelName=pa.Column(str, coerce=True),
#             channel=pa.Column(pa.Int8, coerce=True),
#             polarity=pa.Column(pa.Int8),
#         )
#     )
#     return schema_digital_inputs.validate(digital_inputs_df)


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
    timestamps_colname: str | None = None,  # for version_1
) -> pd.DataFrame:
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
            assert timestamps_colname is not None, 'for version 1, column name for timestamps need to be provided'
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


# def read_digital_inputs_csv(path: str | Path, version=None, validate=True, channel=None) -> pd.DataFrame:
#     df = read_digital_inputs_file(path)
#     if version is None:
#         infer_neurophotometrics_version_from_digital_inputs(df)
#     match version:
#         case 'new':
#             df_digital_inputs = pd.read_csv(path, header=None)
#             df_digital_inputs.columns = [  # FIXME replace with rename
#                 'ChannelName',
#                 'Channel',
#                 'AlwaysTrue',
#                 'SystemTimestamp',
#                 'ComputerTimestamp',
#             ]

#             if validate:
#                 df_digital_inputs = validate_neurophotometrics_digital_inputs(df_digital_inputs, version=version)

#         case 'old':
#             # this is for the deprecated and legacy file format of the neurophotometrics
#             # channel kwarg needs to be provided
#             assert channel is not None, 'if using the legacy digital inputs format, a channel index needs to be provided'
#             df_digital_inputs = pd.read_csv(path).groupby('Value.Value').get_group(True)
#             df_digital_inputs = df_digital_inputs.drop(['Value.Value', 'Value.Seconds'], axis=1)
#             df_digital_inputs['Channel'] = channel
#             df_digital_inputs = df_digital_inputs.rename(columns={'Timestamp': 'SystemTimestamp'})

#             if validate:
#                 df_digital_inputs = validate_neurophotometrics_digital_inputs(df_digital_inputs, version=version)

#         case 'very_old':
#             # this is for the deprecated and legacy file format of the neurophotometrics
#             # channel kwarg needs to be provided
#             assert channel is not None, 'if using the legacy digital inputs format, a channel index needs to be provided'
#             df_digital_inputs = pd.read_csv(path)
#             df_digital_inputs = df_digital_inputs.groupby('Value').get_group(True).drop('Value', axis=1)
#             df_digital_inputs['Channel'] = channel
#             df_digital_inputs.columns = [
#                 'SystemTimestamp',
#                 'Channel',
#             ]
#             if validate:
#                 df_digital_inputs = validate_neurophotometrics_digital_inputs(df_digital_inputs, version=version)

#     return df_digital_inputs


# def validate_neurophotometrics_df(
#     df: pd.DataFrame,
#     data_columns=None,
#     version='infer',
# ) -> pd.DataFrame:
#     if version == 'infer':
#         version = infer_neurophotometrics_version_from_data(df)

#     match version:
#         case 'new':  # kcenia, carolina
#             schema_raw_data = pa.DataFrameSchema(
#                 columns=dict(
#                     FrameCounter=pa.Column(pa.Int64),
#                     SystemTimestamp=pa.Column(pa.Float64),
#                     LedState=pa.Column(pa.Int16, coerce=True),
#                     ComputerTimestamp=pa.Column(pa.Float64),
#                     **{k: pa.Column(pa.Float64) for k in data_columns},
#                 )
#             )

#         case 'old':  # alejandro
#             schema_raw_data = pa.DataFrameSchema(
#                 columns=dict(
#                     FrameCounter=pa.Column(pa.Int64),
#                     Timestamp=pa.Column(pa.Float64),
#                     LedState=pa.Column(pa.Int16, coerce=True),
#                     **{k: pa.Column(pa.Float64) for k in data_columns},
#                 )
#             )

#         case 'very_old':  # also kcenia
#             schema_raw_data = pa.DataFrameSchema(
#                 columns=dict(
#                     FrameCounter=pa.Column(pa.Int64),
#                     Timestamp=pa.Column(pa.Float64),
#                     Flags=pa.Column(pa.Int16, coerce=True),
#                     **{k: pa.Column(pa.Float64) for k in data_columns},
#                 )
#             )

#         case _:
#             raise ValueError(f'unknown version {version}')

#     return schema_raw_data.validate(df)


# def validate_neurophotometrics_digital_inputs(df: pd.DataFrame, version='new') -> pd.DataFrame:
#     match version:
#         case 'new':
#             schema_digital_inputs = pa.DataFrameSchema(
#                 columns=dict(
#                     ChannelName=pa.Column(str, coerce=True),
#                     Channel=pa.Column(pa.Int8, coerce=True),
#                     AlwaysTrue=pa.Column(bool, coerce=True),
#                     SystemTimestamp=pa.Column(pa.Float64),
#                     ComputerTimestamp=pa.Column(pa.Float64),
#                 )
#             )

#         case 'old':
#             schema_digital_inputs = pa.DataFrameSchema(
#                 columns=dict(
#                     Channel=pa.Column(pa.Int8, coerce=True),
#                     SystemTimestamp=pa.Column(pa.Float64),
#                 )
#             )

#         # case 'very_old':
#         #     schema_digital_inputs = pa.DataFrameSchema(
#         #         columns=
#         #     )

#         case _:
#             raise ValueError(f'unknown version {version}')

#     return schema_digital_inputs.validate(df)
