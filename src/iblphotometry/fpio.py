import numpy as np
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from one.api import ONE
from typing import Optional, Dict, List
from dataclasses import field
from brainbox.io.one import SessionLoader
import warnings
from functools import wraps
from iblphotometry import neurophotometrics


"""
##     ##    ###    ##       #### ########     ###    ######## ####  #######  ##    ##
##     ##   ## ##   ##        ##  ##     ##   ## ##      ##     ##  ##     ## ###   ##
##     ##  ##   ##  ##        ##  ##     ##  ##   ##     ##     ##  ##     ## ####  ##
##     ## ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ## ## ##
 ##   ##  ######### ##        ##  ##     ## #########    ##     ##  ##     ## ##  ####
  ## ##   ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ##   ###
   ###    ##     ## ######## #### ########  ##     ##    ##    ####  #######  ##    ##
"""


photometry_df_schema = {
    'times': pa.Column(pa.Float64),
    'valid': pa.Column(pa.Bool),
    'wavelength': pa.Column(pa.Float64, nullable=True),
    'name': pa.Column(pa.String),  # this should rather be "channel_name" or "channel"
    'color': pa.Column(pa.String),
}


def _infer_data_columns(df: pd.DataFrame) -> list[str]:
    # small helper, returns the data columns from a photometry dataframe
    if any([col.startswith('Region') for col in df.columns]):
        data_columns = [col for col in df.columns if col.startswith('Region')]
    else:
        data_columns = [col for col in df.columns if col.startswith('R') or col.startswith('G')]
    return data_columns


def validate_photometry_df(
    photometry_df: pd.DataFrame,
    data_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Validate the photometry DataFrame against the schema.

    Args:
        photometry_df (pd.DataFrame): Input DataFrame.
        data_columns (Optional[List[str]]): List of data columns to validate. If None, inferred automatically.

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


def from_photometry_df(
    photometry_df: pd.DataFrame,
    data_columns: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    rename: Optional[Dict] = None,  # the dict to rename the data_columns -> Region?G | G? -> brain_region
    validate: bool = True,
    drop_first: bool = True,
) -> dict[pd.DataFrame]:
    """
    Split a photometry DataFrame into separate DataFrames per channel.

    Args:
        photometry_df (pd.DataFrame): Input photometry DataFrame.
        data_columns (Optional[List[str]]): List of data columns. If None, inferred automatically.
        channel_names (Optional[List[str]]): List of channel names. If None, inferred automatically.
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
    locations_pqt_path: Optional[str | Path] = None,
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


def from_eid(
    eid: str,
    one: ONE,
    collection: str = 'photometry',
    drop_first: bool = True,
    revision: str | None = None,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Load photometry data for a session ID (eid) using ONE.

    Args:
        eid (str): Session ID.
        one: ONE API instance.

    Returns:
        list[dict]: List of channel DataFrames.
    """
    datasets = ['photometry.signal.pqt', 'photometryROI.locations.pqt']
    for dataset in datasets:
        one.load_dataset(
            eid,
            dataset,
            collection=f'alf/{collection}',
            revision=revision,
            download_only=True,
        )
    return from_session_path(one.eid2path(eid), drop_first=drop_first)


def from_session_path(
    session_path: str | Path,
    collection: str = 'photometry',
    drop_first: bool = True,
    revision: Optional[str] = None,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Load photometry data from a locally present session path.

    Args:
        session_path (str | Path): Path to session folder.
        drop_first (bool): Whether to drop the first frame.

    Returns:
        list[dict]: List of channel DataFrames.
    """
    session_path = Path(session_path) if isinstance(session_path, str) else session_path
    if revision in ['', None]:
        data_paths = [
            session_path / f'alf/{collection}/photometry.signal.pqt',
            session_path / f'alf/{collection}/photometryROI.locations.pqt',
        ]
    else:
        raise NotImplementedError('loading with revisions from a local folder is not implemented yet')

    return from_photometry_pqt(
        *data_paths,
        drop_first=drop_first,
    )


def restrict_to_session_time(
    raw_dfs: list[dict],
    trials_df: pd.DataFrame,
    pre: float = -5.0,
    post: float = 5.0,
):
    t_start = trials_df.iloc[0]['intervals_0']
    t_stop = trials_df.iloc[-1]['intervals_1']

    for band in raw_dfs.keys():
        df = raw_dfs[band]
        ix = np.logical_and(
            df.index.values > t_start + pre,
            df.index.values < t_stop + post,
        )
        raw_dfs[band] = df.loc[ix]

    # the above indexing can lead to unevenly shaped bands.
    # Cut to shortest
    n = np.min([df.shape[0] for _, df in raw_dfs.items()])
    for band in raw_dfs.keys():
        raw_dfs[band] = raw_dfs[band].iloc[:n]

    return raw_dfs


"""
 
  ######  ########  ######   ######  ####  #######  ##    ## ##        #######     ###    ########  ######## ########  
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ###   ## ##       ##     ##   ## ##   ##     ## ##       ##     ## 
 ##       ##       ##       ##        ##  ##     ## ####  ## ##       ##     ##  ##   ##  ##     ## ##       ##     ## 
  ######  ######    ######   ######   ##  ##     ## ## ## ## ##       ##     ## ##     ## ##     ## ######   ########  
       ## ##             ##       ##  ##  ##     ## ##  #### ##       ##     ## ######### ##     ## ##       ##   ##   
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ##   ### ##       ##     ## ##     ## ##     ## ##       ##    ##  
  ######  ########  ######   ######  ####  #######  ##    ## ########  #######  ##     ## ########  ######## ##     ## 
 
"""


class PhotometrySessionLoader(SessionLoader):
    photometry: dict = field(default_factory=dict, repr=False)

    def __init__(self, *args, photometry_collection: str = 'photometry', **kwargs):
        self.photometry_collection = photometry_collection
        self.revision = kwargs.get('revision', None)

        # determine if loading by eid or session path
        self.load_by_path = True if 'session_path' in kwargs else False

        super().__init__(*args, **kwargs)

    def load_session_data(self, **kwargs):
        super().load_session_data(**kwargs)
        self.load_photometry()

    def load_photometry(
        self,
        restrict_to_session: bool = True,
        pre: int = -5,
        post: int = 5,
    ):
        # session path precedence over eid
        if self.load_by_path:
            raw_dfs = from_session_path(
                self.session_path,
                collection=self.photometry_collection,
                revision=self.revision,
            )
        else:  # load by eid
            raw_dfs = from_eid(
                self.eid,
                self.one,
                collection=self.photometry_collection,
                revision=self.revision,
            )

        if restrict_to_session:
            if isinstance(self.trials, pd.DataFrame) and (self.trials.shape[0] == 0):
                self.load_trials()
            raw_dfs = restrict_to_session_time(raw_dfs, self.trials, pre, post)

        self.photometry = raw_dfs


def _deprecated_forward(target_func):
    """
    Return a wrapper that warns once and calls target_func.
    Keeps __name__, __doc__ and signature via functools.wraps.
    """

    @wraps(target_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f'{target_func.__module__}.{target_func.__name__} is deprecated; '
            f'use {target_func.__module__}.{target_func.__name__} from the new module instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return target_func(*args, **kwargs)

    return wrapper


# moved functions


infer_neurophotometrics_version_from_data = _deprecated_forward(neurophotometrics.infer_neurophotometrics_version_from_data)
read_neurophotometrics_file = _deprecated_forward(neurophotometrics.read_neurophotometrics_file)
from_neurophotometrics_df_to_photometry_df = _deprecated_forward(neurophotometrics.from_neurophotometrics_df_to_photometry_df)
from_neurophotometrics_file_to_photometry_df = _deprecated_forward(neurophotometrics.from_neurophotometrics_file_to_photometry_df)
from_neurophotometrics_file = _deprecated_forward(neurophotometrics.from_neurophotometrics_file)
infer_neurophotometrics_version_from_digital_inputs = _deprecated_forward(
    neurophotometrics.infer_neurophotometrics_version_from_digital_inputs
)
read_digital_inputs_file = _deprecated_forward(neurophotometrics.read_digital_inputs_file)
validate_digital_inputs_df = _deprecated_forward(neurophotometrics.validate_digital_inputs_df)
