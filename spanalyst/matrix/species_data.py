"""Matrix to summarize 2 dimensions of data by counts of a third in a sparse matrix."""
import json
from logging import ERROR
from numpy import integer as np_int, floating as np_float, ndarray
import os
from zipfile import ZipFile

from spanalyst.common.constants import (
    JSON_EXTENSION, SNKeys, SUMMARY, TMP_PATH, ZIP_EXTENSION
)
from spanalyst.common.log import logit


# .............................................................................
class _SpeciesDataMatrix:
    """Class for managing computations for counts of species x aggregator."""

    # ...........................
    def __init__(self, dim0, dim1, table_type, datestr, logger=None):
        """Constructor for species by dataset comparisons.

        Args:
            dim0 (specnet.common.constants.ANALYSIS_DIM: dimension for axis 0, rows
            dim1 (specnet.common.constants.ANALYSIS_DIM: dimension for axis 1, columns
            table_type (code from spanalyst.common.constants.SUMMARY): predefined type of
                data indicating type and contents.
            datestr (str): date of the source data in YYYY_MM_DD format.
            logger (object): An optional local logger to use for logging output
                with consistent options

        Raises:
            Exception: on invalid table_type.

        Note: in the first implementation, because species are generally far more
            numerous, rows are always species, columns are datasets.  This allows
            easier exporting to other formats (i.e. Excel), which allows more rows than
            columns.

        Note:
            All filenames have the same basename with extensions indicating which data
                they contain. The filename contains a string like YYYY-MM-DD which
                indicates which GBIF data dump the statistics were built upon.
        """
        # Test that table type agrees with provided dimensions
        _, dim0code, dim1code, _ = SUMMARY.parse_table_type(table_type)
        if dim0["code"] != dim0code:
            raise Exception(f"Dimension 0 {dim0['code']} != {dim0code} from {table_type}")
        if dim1["code"] != dim1code:
            raise Exception(f"Dimension 1 {dim1['code']} != {dim1code} from {table_type}")

        self._row_dim = dim0
        self._col_dim = dim1
        self._table_type = table_type
        self._datestr = datestr

        try:
            self._table = SUMMARY.get_table(table_type, datestr=datestr)
        except Exception as e:
            raise Exception(f"Cannot create _AggregateDataMatrix: {e}")

        self._keys = SNKeys.get_keys_for_table(table_type)
        self._logger = logger
        self._report = {}

    # ...........................
    @property
    def table_type(self):
        return self._table_type

    # ...........................
    @property
    def datestr(self):
        return self._datestr

    # ...........................
    @property
    def y_dimension(self):
        """Return analysis dimension for axis 0.

        Returns:
        (specnet.common.constants.ANALYSIS_DIM): Data dimension for axis 0 (rows).
        """
        return self._row_dim

    # ...........................
    @property
    def x_dimension(self):
        """Return analysis dimension for axis 1.

        Returns:
        (specnete.common.constants.ANALYSIS_DIM): Data dimension for axis 1 (columns).
        """
        return self._col_dim

    # ...........................
    @property
    def dimensions(self):
        """Return analysis dimension for axis 1.

        Returns:
            (specnet.common.constants.ANALYSIS_DIM): Data dimension for axis 1 (columns).
        """
        return (self._row_dim["code"], self._col_dim["code"])

    # ...............................................
    def _logme(self, msg, refname="", log_level=None):
        logit(msg, logger=self._logger, refname=refname, log_level=log_level)

    # ...............................................
    @classmethod
    def get_matrix_meta_zip_filenames(cls, table, local_path=None):
        """Return the files that comprise local input data, optionally delete.

        Args:
            table (dict): dictionary of metadata for a matrix
            local_path (str): Absolute path of local destination path

        Returns:
            mtx_fname (str): absolute path for local matrix data file.
            meta_fname (str): absolute path for local metadata file.
            zip_fname (str): absolute path for local compressed file.

        Note:
            Leave local_path as None to return S3 object names.
        """
        basename = table["fname"]
        mtx_ext = table["file_extension"]
        mtx_fname = f"{basename}{mtx_ext}"
        meta_fname = f"{basename}{JSON_EXTENSION}"
        zip_fname = f"{basename}{ZIP_EXTENSION}"
        if local_path is not None:
            mtx_fname = os.path.join(local_path, mtx_fname)
            meta_fname = os.path.join(local_path, meta_fname)
            zip_fname = os.path.join(local_path, zip_fname)
        return mtx_fname, meta_fname, zip_fname

    # ......................................................
    @staticmethod
    def convert_np_vals_for_json(obj):
        """Encode numpy values (from matrix operations) for JSON output.

        Args:
            obj: a simple numpy object, value or array

        Returns:
            an object serializable by JSON

        Note:
            from https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
        """
        if isinstance(obj, np_int):
            return int(obj)
        elif isinstance(obj, np_float):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        else:
            return obj

    # ...............................................
    @classmethod
    def _dump_metadata(self, metadata, meta_fname):
        """Write metadata to a json file, deleting it first if exists.

        Args:
            metadata (dict): metadata about matrix
            meta_fname (str): local output filename for JSON metadata.

        Raises:
            Exception: on failure to serialize metadata as JSON.
            Exception: on failure to write metadata json string to file.
        """
        if os.path.exists(meta_fname):
            os.remove(meta_fname)
            print(f"Removed file {meta_fname}.")

        try:
            metastr = json.dumps(metadata)
        except Exception as e:
            raise Exception(f"Failed to serialize metadata as JSON: {e}")
        try:
            with open(meta_fname, 'w') as outf:
                outf.write(metastr)
        except Exception as e:
            raise Exception(f"Failed to write metadata to {meta_fname}: {e}")

    # ...............................................
    @classmethod
    def load_metadata(cls, meta_filename):
        """Read JSON metadata for a matrix.

        Args:
            meta_filename (str): Filename of metadata to read .

        Returns:
            meta_dict (dict): metadata for a matrix

        Raises:
            Exception: on failure to read file.
            Exception: on failure load JSON metadata into a dictionary
        """
        # Read JSON dictionary as string
        try:
            with open(meta_filename) as metaf:
                meta_str = metaf.read()
        except Exception as e:
            raise Exception(f"Failed to load {meta_filename}: {e}")
        # Load metadata from string
        try:
            meta_dict = json.loads(meta_str)
        except Exception as e:
            raise Exception(f"Failed to load {meta_filename}: {e}")

        return meta_dict

    # ...............................................
    @classmethod
    def _find_local_files(cls, expected_files):
        # Are local files already present?
        files_present = [fname for fname in expected_files if os.path.exists(fname)]
        all_exist = len(files_present) == len(expected_files)
        return files_present, all_exist

    # ...............................................
    def _compress_files(self, input_fnames, zip_fname):
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
            self._logme(f"Removed file {zip_fname}.")

        try:
            with ZipFile(zip_fname, 'w') as zip:
                for fname in input_fnames:
                    zip.write(fname, os.path.basename(fname))
        except Exception as e:
            msg = f"Failed to write {zip_fname}: {e}"
            self._logme(msg, log_level=ERROR)
            raise Exception(msg)

    # .............................................................................
    def _remove_expected_files(self, local_path=TMP_PATH):
        # Always delete local files before compressing this data.
        expected_files = self.get_matrix_meta_zip_filenames(
            self._table, local_path=local_path)
        for fn in expected_files:
            if os.path.exists(fn):
                os.remove(fn)
        return expected_files

    # .............................................................................
    @classmethod
    def _uncompress_files(cls, zip_filename, local_path, overwrite=False):
        """Uncompress a zipped SparseMatrix into a coo_array and row/column categories.

        Args:
            zip_filename (str): Filename of output data to write to S3.
            local_path (str): Absolute path of local destination path
            overwrite (bool): Flag indicating whether to use or delete existing files
                prior to unzipping.

        Returns:
            sparse_coo (scipy.sparse.coo_array): Sparse Matrix containing data.
            row_categ (pandas.api.types.CategoricalDtype): row categories
            col_categ (pandas.api.types.CategoricalDtype): column categories
            table_type (aws.aws_constants.SUMMARY_TABLE_TYPES): type of table data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on missing input zipfile
            Exception: on failure to parse filename
            Exception: on missing expected file from zipfile
        """
        if not os.path.exists(zip_filename):
            raise Exception(f"Missing file {zip_filename}")
        try:
            table_type, datestr = SUMMARY.get_tabletype_datestring_from_filename(
                zip_filename)
        except Exception:
            raise

        table = SUMMARY.get_table(table_type, datestr=datestr)
        mtx_fname, meta_fname, _ = cls.get_matrix_meta_zip_filenames(
            table, local_path=local_path)

        # Are local files already present?
        expected_files = [mtx_fname, meta_fname]
        files_present, all_exist = cls._find_local_files(expected_files)

        # If overwrite or only some are present, remove
        if (
                len(files_present) > 0 and
                (overwrite is True or
                 (overwrite is False and all_exist is False))
        ):
            for fn in files_present:
                if os.path.exists(fn):
                    os.remove(fn)

        if all_exist and overwrite is False:
            print(f"Expected files {', '.join(expected_files)} already exist.")
        else:
            # Unzip to local dir
            with ZipFile(zip_filename, mode="r") as archive:
                archive.extractall(f"{local_path}/")
            for fn in [mtx_fname, meta_fname]:
                if not os.path.exists(fn):
                    raise Exception(f"Missing expected file {fn}")

        return mtx_fname, meta_fname, table_type, datestr
