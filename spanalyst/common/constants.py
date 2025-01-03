"""Constants for spanalyst module for analyzing GBIF species occurrence data dimensions."""
from copy import deepcopy
from enum import Enum
import os

from spanalyst.aws.constants import S3_RS_TABLE_SUFFIX


# .............................................................................
# Data constants
# .............................................................................
SPECIES_DIMENSION = "species"
COMPOUND_SPECIES_FLD = "taxonkey_species"
GBIF_DATASET_KEY_FLD = "dataset_key"

# .............................................................................
# Log processing progress
LOGINTERVAL = 1000000
LOG_FORMAT = " ".join(["%(asctime)s", "%(levelname)-8s", "%(message)s"])
LOG_DATE_FORMAT = "%d %b %Y %H:%M"
LOGFILE_MAX_BYTES = 52000000
LOGFILE_BACKUP_COUNT = 5

TMP_PATH = "/tmp"
ENCODING = "utf-8"
ERR_SEPARATOR = "------------"
USER_DATA_TOKEN = "###SCRIPT_GOES_HERE###"
CSV_DELIMITER = ","
ZIP_EXTENSION = ".zip"
JSON_EXTENSION = ".json"
CSV_EXTENSION = ".csv"

SHP_EXT = "shp"
SHP_EXTENSIONS = [
    ".shp", ".shx", ".dbf", ".prj", ".sbn", ".sbx", ".fbn", ".fbx", ".ain",
    ".aih", ".ixs", ".mxs", ".atx", ".shp.xml", ".cpg", ".qix"]


# .............................................................................
class SUMMARY_FIELDS:
    """Fields used to summarize aggregated data."""
    COUNT = "count"
    TOTAL = "total"
    OCCURRENCE_COUNT = "occ_count"
    SPECIES_COUNT = "species_count"


# .............................................................................
class STATISTICS_TYPE:
    """Biodiversity statistics for a Site by Species presence-absence matrix (PAM)."""
    SIGMA_SITE = "sigma-site"
    SIGMA_SPECIES = "sigma-species"
    DIVERSITY = "diversity"
    SITE = "site"
    SPECIES = "species"

# ...........................
    @classmethod
    def all(cls):
        """Get all aggregated data type codes.

        Returns:
            list of supported codes for datatypes.
        """
        return (cls.SIGMA_SITE, cls.SIGMA_SPECIES, cls.DIVERSITY, cls.SITE, cls.SPECIES)


# .............................................................................
class AGGREGATION_TYPE:
    """Types of tables created for aggregate species data analyses."""
    # TODO: decide whether to keep PAM
    LIST = "list"
    COUNT = "counts"
    MATRIX = "matrix"
    PAM = "pam"
    STATISTICS = "stats"
    SUMMARY = "summary"

    # ...........................
    @classmethod
    def all(cls):
        """Get all aggregated data type codes.

        Returns:
            list of supported codes for datatypes.
        """
        return (cls.LIST, cls.COUNT, cls.MATRIX, cls.PAM, cls.STATISTICS, cls.SUMMARY)


# .............................................................................
class SNKeys(Enum):
    """Dictionary keys to use for describing RowColumnComparisons of SUMMARY data.

    Note: All keys refer to the relationship between rows, columns and values.  Missing
        values in a dataset dictionary indicate that the measure is not meaningful.
    """
    # ----------------------------------------------------------------------
    # Column: type of aggregation
    (COL_TYPE,) = range(5000, 5001)
    # Column: One x
    (COL_LABEL, COL_COUNT, COL_TOTAL,
     COL_MIN_TOTAL, COL_MIN_TOTAL_NUMBER, COL_MAX_TOTAL, COL_MAX_TOTAL_LABELS,
     ) = range(5100, 5107)
    # Column: All x
    (COLS_TOTAL,
     COLS_MIN_TOTAL, COLS_MIN_TOTAL_NUMBER, COLS_MEAN_TOTAL, COLS_MEDIAN_TOTAL,
     COLS_MAX_TOTAL, COLS_MAX_TOTAL_LABELS,
     COLS_COUNT,
     COLS_MIN_COUNT, COLS_MIN_COUNT_NUMBER, COLS_MEAN_COUNT, COLS_MEDIAN_COUNT,
     COLS_MAX_COUNT, COLS_MAX_COUNT_LABELS
     ) = range(5200, 5214)
    # Row: aggregation of what type of data
    (ROW_TYPE,) = range(6000, 6001)
    # Row: One y
    (ROW_LABEL, ROW_COUNT, ROW_TOTAL,
     ROW_MIN_TOTAL, ROW_MIN_TOTAL_NUMBER, ROW_MAX_TOTAL, ROW_MAX_TOTAL_LABELS,
     ) = range(6100, 6107)
    # Rows: All y
    (ROWS_TOTAL,
     ROWS_MIN_TOTAL, ROWS_MIN_TOTAL_NUMBER, ROWS_MEAN_TOTAL, ROWS_MEDIAN_TOTAL,
     ROWS_MAX_TOTAL, ROWS_MAX_TOTAL_LABELS,
     ROWS_COUNT,
     ROWS_MIN_COUNT, ROWS_MIN_COUNT_NUMBER, ROWS_MEAN_COUNT, ROWS_MEDIAN_COUNT,
     ROWS_MAX_COUNT, ROWS_MAX_COUNT_LABELS
     ) = range(6200, 6214)
    # Type of aggregation
    (TYPE,) = range(0, 1)
    # One field of row/column header
    (ONE_LABEL, ONE_COUNT, ONE_TOTAL,
     ONE_MIN_COUNT, ONE_MIN_COUNT_NUMBER,
     ONE_MAX_COUNT, ONE_MAX_COUNT_LABELS
     ) = range(100, 107)
    # Column: All row/column headers
    (ALL_TOTAL,
     ALL_MIN_TOTAL, ALL_MIN_TOTAL_NUMBER, ALL_MEAN_TOTAL, ALL_MEDIAN_TOTAL,
     ALL_MAX_TOTAL, ALL_MAX_TOTAL_LABELS,
     ALL_COUNT,
     ALL_MIN_COUNT, ALL_MIN_COUNT_NUMBER, ALL_MEAN_COUNT, ALL_MEDIAN_COUNT,
     ALL_MAX_COUNT, ALL_MAX_COUNT_LABELS,
     ) = range(200, 214)

    @classmethod
    def get_keys_for_table(cls, table_type, dim_other):
        """Return keystrings for statistics dictionary for specific aggregation tables.

        Args:
            table_type (aws_constants.SUMMARY_TABLE_TYPES): type of aggregated data

        Returns:
            keys (dict): Dictionary of strings to be used as keys for each type of
                value in a dictionary of statistics.

        Raises:
            Exception: on axis/dimension 1 not equal to "species".
        """
        datacontents, dim0, dim1, datatype = SUMMARY.parse_table_type(table_type)
        dim_species = SPECIES_DIMENSION
        keys = {}
        if datatype == AGGREGATION_TYPE.MATRIX:
            # dim1 is always species in matrix
            if not(dim0 == dim_other and dim1 == dim_species):
                raise Exception(
                    f"Invalid tabletype {table_type}. Matrix axis 1 must be species.")
            else:
                keys = {
                    # ----------------------------------------------------------------------
                    # Column
                    # -----------------------------
                    cls.COL_TYPE: dim_other,
                    # One dataset
                    cls.COL_LABEL: f"{dim_other}_label",
                    # Count (non-zero elements in column)
                    cls.COL_COUNT: f"total_species_for_{dim_other}",
                    # Values (total of values in column)
                    cls.COL_TOTAL: f"total_occurrences_for_{dim_other}",
                    # Values: Minimum occurrences for one dataset, species labels
                    cls.COL_MIN_TOTAL: f"min_occurrences_for_{dim_other}",
                    cls.COL_MIN_TOTAL_NUMBER: f"number_of_species_with_min_occurrences_for_{dim_other}",
                    # Values: Maximum occurrence count for one dataset, species labels
                    cls.COL_MAX_TOTAL: f"max_occurrences_for_{dim_other}",
                    cls.COL_MAX_TOTAL_LABELS: f"species_with_max_occurrences_for_{dim_other}",

                    # All datasets
                    # ------------
                    # Values: Total of all occurrences for all datasets - stats
                    cls.COLS_TOTAL: f"total_occurrences_of_all_{dim_other}s",
                    cls.COLS_MIN_TOTAL: f"min_occurrences_of_all_{dim_other}s",
                    cls.COLS_MIN_TOTAL_NUMBER: f"number_of_{dim_other}s_with_min_occurrences_of_all",
                    cls.COLS_MEAN_TOTAL: f"mean_occurrences_of_all_{dim_other}s",
                    cls.COLS_MEDIAN_TOTAL: f"median_occurrences_of_all_{dim_other}s",
                    cls.COLS_MAX_TOTAL: f"max_occurrences_of_all_{dim_other}s",
                    cls.COLS_MAX_TOTAL_LABELS: f"{dim_other}s_with_max_occurrences_of_all",
                    # ------------
                    # Counts: Count of all species (from all columns/datasets)
                    cls.COLS_COUNT: f"total_{dim_other}_count",
                    # Species counts for all datasets - stats
                    cls.COLS_MIN_COUNT: f"min_species_count_of_all_{dim_other}s",
                    cls.COLS_MIN_COUNT_NUMBER: f"number_of_{dim_other}s_with_min_species_count_of_all",
                    cls.COLS_MEAN_COUNT: f"mean_species_count_of_all_{dim_other}s",
                    cls.COLS_MEDIAN_COUNT: f"median_species_count_of_all_{dim_other}s",
                    cls.COLS_MAX_COUNT: f"max_species_count_of_all_{dim_other}s",
                    cls.COLS_MAX_COUNT_LABELS: f"{dim_other}s_with_max_species_count_of_all",
                    # ----------------------------------------------------------------------
                    # Row
                    # -----------------------------
                    cls.ROW_TYPE: dim1,
                    # One species
                    cls.ROW_LABEL: f"species_label",
                    # Count (non-zero elements in row)
                    cls.ROW_COUNT: f"total_{dim_other}s_for_species",
                    # Values (total of values in row)
                    cls.ROW_TOTAL: f"total_occurrences_for_species",
                    # Values: Minimum occurrence count for one species, otherdim labels, indexes
                    cls.ROW_MIN_TOTAL: f"min_occurrences_for_species",
                    # Values: Maximum occurrence count for one species, otherdim labels, indexes
                    cls.ROW_MAX_TOTAL: f"max_occurrences_for_species",
                    cls.ROW_MAX_TOTAL_LABELS: f"{dim_other}s_with_max_occurrences_for_species",
                    # -----------------------------
                    # All species
                    # ------------
                    # COMPARES TO: cls.ROW_TOTAL: "total_occurrences_for_species",
                    # Values: Total of all occurrences for all species - stats
                    cls.ROWS_TOTAL: "total_occurrences_of_all_species",
                    cls.ROWS_MIN_TOTAL: "min_occurrences_of_all_species",
                    cls.ROWS_MIN_TOTAL_NUMBER: "number_of_species_with_max_occurrences_of_all",
                    cls.ROWS_MEAN_TOTAL: "mean_occurrences_of_all_species",
                    cls.ROWS_MEDIAN_TOTAL: "median_occurrences_of_all_species",
                    cls.ROWS_MAX_TOTAL: "max_occurrences_of_all_species",
                    cls.ROWS_MAX_TOTAL_LABELS: "species_with_max_occurrences_of_all",
                    # ------------
                    # COMPARES TO: cls.ROW_COUNT: "total_datasets_for_species",
                    # Counts: Count of all datasets (from all rows/species)
                    cls.ROWS_COUNT: "total_species_count",
                    # Dataset counts for all species - stats
                    cls.ROWS_MIN_COUNT: f"min_{dim_other}_count_of_all_species",
                    cls.ROWS_MIN_COUNT_NUMBER: f"species_with_min_{dim_other}_count_of_all",
                    cls.ROWS_MEAN_COUNT: f"mean_{dim_other}_count_of_all_species",
                    cls.ROWS_MEDIAN_COUNT: f"median_{dim_other}_count_of_all_species",
                    cls.ROWS_MAX_COUNT: f"max_{dim_other}_count_of_all_species",
                    cls.ROWS_MAX_COUNT_LABELS: f"species_with_max_{dim_other}_count_of_all",
                }
        elif datatype == AGGREGATION_TYPE.SUMMARY:
            keys = {
                # ----------------------------------------------------------------------
                # Column
                # -----------------------------
                cls.TYPE: dim0,
                # One dataset
                cls.ONE_LABEL: f"{dim0}_label",
                # Count (non-zero elements in column)
                cls.ONE_COUNT: f"total_{dim1}_for_{dim0}",
                # Values (total of values in column)
                cls.ONE_TOTAL: f"total_occurrences_for_{dim0}",
                # Values: Minimum occurrence count for one {dim0}
                cls.ONE_MIN_COUNT: f"min_occurrences_for_{dim0}",
                cls.ONE_MIN_COUNT_NUMBER: f"number_of_{dim0}s_with_min_occurrences",
                # Values: Maximum occurrence count for one dataset, species labels, indexes
                cls.ONE_MAX_COUNT: f"max_occurrences_for_{dim0}",
                cls.ONE_MAX_COUNT_LABELS: f"{dim0}s_with_max_occurrences",
                # -----------------------------
                # All datasets
                # ------------
                # COMPARES TO:  cls.ONE_TOTAL: "total_occurrences_for_{dim0}",
                # Values: Total of all occurrences for all otherdim - stats
                cls.ALL_TOTAL: f"total_occurrences_of_all_{dim0}s",
                cls.ALL_MIN_TOTAL: f"min_occurrences_of_all_{dim0}s",
                cls.ALL_MIN_TOTAL_NUMBER: f"number_of_{dim0}s_with_min_occurrences_of_all",
                cls.ALL_MEAN_TOTAL: f"mean_occurrences_of_all_{dim0}s",
                cls.ALL_MEDIAN_TOTAL: f"median_occurrences_of_all_{dim0}s",
                cls.ALL_MAX_TOTAL: f"max_occurrences_of_all_{dim0}s",
                # ------------
                # COMPARES TO: cls.ONE_COUNT: "total_{dim1}_for_{dim0}",
                # Counts: Count of all species (from all columns/datasets)
                cls.ALL_COUNT: "total_{dim1}_count",
                # Species counts for all datasets - stats
                cls.ALL_MIN_COUNT: f"min_{dim1}_count_of_all_{dim0}s",
                cls.ALL_MEAN_COUNT: f"mean_{dim1}_count_of_all_{dim0}s",
                cls.ALL_MEDIAN_COUNT: f"median_{dim1}_count_of_all_{dim0}s",
                cls.ALL_MAX_COUNT: f"max_{dim1}_count_of_all_{dim0}s",
            }
        if len(keys) == 0:
            for k, v in cls.__members__.items():
                keys[v] = k.lower()
        return keys


# .............................................................................
class ANALYSIS_DIM:
    """All dimensions with input data columns used for data analyses."""
    SPECIES = {
        "code": "species",
        "key_fld": COMPOUND_SPECIES_FLD,
    }

    # ...........................
    def __init__(self, other_dim_dicts):
        """Constructor for project-specific analyses.

        Args:
            other_dim_dicts (list of dicts): each dictionary contains keys "code" and
                "key_fld" with values the dimension code and input data key fieldname.
                Data is aggregated on values in the key field.

        Raises:
            Exception: on wrong parameter type.
        """
        if not type(other_dim_dicts) is list:
            if type(other_dim_dicts) is dict:
                other_dim_dicts = [other_dim_dicts]
            else:
                raise Exception(f"Parameter {other_dim_dicts} must be a list of dicts.")
        self.DIMENSIONS = []
        for d in other_dim_dicts:
            dim = {"code": d["code"], "key_fld": d["key_fld"]}
            self.DIMENSIONS.append(dim)

    # ...........................
    def species(self):
        """Get the data species analyses dimension.

        Returns:
            Data dimension relating to species.
        """
        return self.SPECIES

    # ...........................
    def species_code(self):
        """Get the code for the data species analyses dimension.

        Returns:
            Code for the data dimension relating to species.
        """
        return self.SPECIES["code"]

    # ...........................
    def analysis_dimensions(self):
        """Get one or all data analyses dimensions to be analyzed for species.

        Returns:
            dim_lst (list): List of data dimension(s) to be analyzed for species.
        """
        return self.DIMENSIONS

    # ...........................
    def analysis_codes(self):
        """Get one or all codes for data analyses dimensions to be analyzed for species.

        Returns:
            code_lst (list): Codes of data dimension(s) to be analyzed for species.
        """
        code_lst = [dim["code"] for dim in self.DIMENSIONS]
        return code_lst

    # ...........................
    def get(self, code):
        """Get the data analyses dimension for the code.

        Args:
            code (str): Code for the analysis dimension to be returned.

        Returns:
            Data dimension.

        Raises:
            Exception: on unknown code.
        """
        for dim in self.DIMENSIONS:
            if code == dim["code"]:
                return dim
        raise Exception(f"No dimension `{code}` in ANALYSIS_DIM")

    # ...........................
    def get_from_key_fld(self, key_fld):
        """Get the data analyses dimension for the key_fld.

        Args:
            key_fld (str): Field name for the analysis dimension to be returned.

        Returns:
            Data dimension.

        Raises:
            Exception: on unknown code.
        """
        for dim in self.DIMENSIONS:
            if key_fld == dim["key_fld"]:
                return dim
        if key_fld == self.SPECIES["key_fld"]:
            return self.SPECIES
        raise Exception(f"No dimension for field `{key_fld}` in ANALYSIS_DIM")


# .............................................................................
class SUMMARY:
    """Types of tables stored in S3 for aggregate species data analyses."""
    dt_token = "YYYY_MM_DD"
    sep = "_"
    dim_sep = f"{sep}x{sep}"
    DATATYPES = AGGREGATION_TYPE.all()
    species_dim = SPECIES_DIMENSION
    species_fld = COMPOUND_SPECIES_FLD

    # ...........................
    def __init__(self, other_dims):
        """Constructor for species by dataset comparisons.

        Args:
            other_dims (list of str): codes for (non-species) dimensions of analysis

        Raises:
            Exception: on wrong parameter type.
        """
        if type(other_dims) is str:
            other_dims = [other_dims]
        else:
            raise Exception(f"Parameter {other_dims} must be a list of strings.")
        self.other_dims = other_dims

    # ...........................
    def get_table_type(self, datatype, dim0, dim1):
        """Get the table_type string for the analysis dimension and datatype.

        Args:
            datatype (SUMMARY.DATAYPES): type of aggregated data.
            dim0 (str): code for primary dimension (bison.common.constants.ANALYSIS_DIM)
                of analysis
            dim1 (str): code for secondary dimension of analysis

        Note:
            BISON Table types include:
                list: region_x_species_list
                counts: region_counts
                summary: region_x_species_summary
                         species_x_region_summary
                matrix:  species_x_region_matrix

        Note: for matrix, dimension1 corresponds to Axis 0 (rows) and dimension2
            corresponds to Axis 1 (columns).

        Returns:
            table_type (str): code for data type and contents

        Raises:
            Exception: on datatype not one of: "counts", "list", "summary", "matrix"
            Exception: on datatype "counts", dim0 not in ANALYSIS_DIMENSIONS
            Exception: on datatype "counts", dim1 not None
            Exception: on datatype "matrix" or "list", dim0 not in ANALYSIS_DIMENSIONS
            Exception: on datatype "matrix" or "list", dim1 != ANALYSIS_DIMENSIONS.SPECIES
            Exception: on dim0 == SPECIES_DIMENSION and dim1 not in ANALYSIS_DIMENSIONS
            Exception: on dim0 in ANALYSIS_DIMENSIONS and dim1 != SPECIES_DIMENSION
        """
        if datatype not in self.DATATYPES:
            raise Exception(f"Datatype {datatype} is not in {self.DATATYPES}.")

        if datatype == AGGREGATION_TYPE.COUNT:
            if dim0 in self.other_dims:
                if dim1 is None:
                    # ex: state_counts
                    table_type = f"{dim0}{self.sep}{datatype}"
                else:
                    raise Exception("Second dimension must be None")
            else:
                raise Exception(
                    f"First dimension for counts must be in {self.other_dims}.")
        elif datatype in (AGGREGATION_TYPE.LIST, AGGREGATION_TYPE.MATRIX):
            if dim0 not in self.other_dims:
                raise Exception(
                    f"First dimension (rows) must be in {self.other_dims}"
                )
            if dim1 != self.species_dim:
                raise Exception(
                    f"Second dimension (columns) must be {self.species_dim}"
                )
            table_type = f"{dim0}{self.dim_sep}{dim1}{self.sep}{datatype}"
        else:
            if dim0 == self.species_dim and dim1 not in self.other_dims:
                raise Exception(
                    f"Second dimension must be in {self.other_dims}"
                )
            elif dim0 in self.other_dims and dim1 != self.species_dim:
                raise Exception(
                    f"First dimension must be {self.species_dim} or "
                    f"in {self.other_dims}."
                )

            table_type = f"{dim0}{self.dim_sep}{dim1}{self.sep}{datatype}"
        return table_type

    # ...........................
    def list(self):
        """Records of dimension, species, occ count for each dimension in project.

        Returns:
            list (dict): dict of dictionaries for each list table defined by the
                project.

        Note:
            The keys for the dictionary (and code in the metadata values) are table_type
        """
        list = {}
        for analysis_code in self.other_dims:
            table_type = self.get_table_type(
                AGGREGATION_TYPE.LIST, analysis_code, self.species_dim)
            dim = ANALYSIS_DIM.get(analysis_code)
            # name == table_type, ex: county_x_species_list
            meta = {
                "code": table_type,
                "fname": f"{table_type}{self.sep}{self.dt_token}",
                # "table_format": "Parquet",
                "file_extension": f"{self.sep}000.parquet",

                "fields": dim["fields"],
                "key_fld": dim["key_fld"],
                "species_fld": self.species_fld,
                "value_fld": SUMMARY_FIELDS.OCCURRENCE_COUNT
            }
            list[table_type] = meta
        return list

    # ...........................
    def counts(self):
        """Records of dimension, species count, occ count for each dimension in project.

        Returns:
            list (dict): dict of dictionaries for each list table defined by the
                project.

        Note:
            This table type refers to a table assembled from original data records
            and species and occurrence counts for a dimension.  The dimension can be
            any unique set of attributes, such as county + riis_status.  For simplicity,
            define each unique set of attributes as a single field/value

        Note:
            The keys for the dictionary (and code in the metadata values) are table_type

        TODO: Remove this from creation?  Can create from sparse matrix.
        """
        counts = {}
        for analysis_code in self.DIMENSIONS:
            dim0 = ANALYSIS_DIM.get(analysis_code)
            table_type = self.get_table_type(
                AGGREGATION_TYPE.COUNT, analysis_code, None)

            meta = {
                "code": table_type,
                "fname": f"{table_type}{self.sep}{self.dt_token}",
                # "table_format": "Parquet",
                "file_extension": f"{self.sep}000.parquet",
                "data_type": "counts",

                # Dimensions: 0 is row (aka Axis 0) list of records with counts
                #   1 is column (aka Axis 1), count and total of dim for each row
                "dim_0_code": analysis_code,
                "dim_1_code": None,

                "key_fld": dim0["key_fld"],
                "occurrence_count_fld": SUMMARY_FIELDS.OCCURRENCE_COUNT,
                "species_count_fld": COMPOUND_SPECIES_FLD,
                "fields": [
                    dim0["key_fld"],
                    SUMMARY_FIELDS.OCCURRENCE_COUNT,
                    COMPOUND_SPECIES_FLD
                ]
            }
            counts[table_type] = meta
        return counts

    # ...........................
    def summary(self):
        """Summary of dimension1 count and occurrence count for each dimension0 value.

        Returns:
            sums (dict): dict of dictionaries for each summary table defined by the
                project.

        Note:
            table contains stacked records summarizing original data:
                dim0, dim1, rec count of dim1 in dim0
                ex: county, species, occ_count
        """
        sums = {}
        species_code = self.species_dim
        for analysis_code in self.other_dims:
            for dim0, dim1 in (
                    (analysis_code, species_code), (species_code, analysis_code)
            ):
                table_type = self.get_table_type(
                    AGGREGATION_TYPE.SUMMARY, dim0, dim1)
                meta = {
                    "code": table_type,
                    "fname": f"{table_type}{self.sep}{self.dt_token}",
                    "file_extension": ".csv",
                    "data_type": "summary",

                    # Dimensions: 0 is row (aka Axis 0) list of values to summarize,
                    #   1 is column (aka Axis 1), count and total of dim for each row
                    "dim_0_code": dim0,
                    "dim_1_code": dim1,

                    # Axis 1
                    "column": "measurement_type",
                    "fields": [SUMMARY_FIELDS.COUNT, SUMMARY_FIELDS.TOTAL],
                    # Matrix values
                    "value": "measure"}
                sums[table_type] = meta
        return sums

    # ...........................
    def matrix(self):
        """Species by <dimension> matrix defined for this project.

        Returns:
            mtxs (dict): dict of dictionaries for each matrix/table defined for this
                project.

        Note:
            Similar to a Presence/Absence Matrix (PAM),
                Rows will always have analysis dimension (i.e. region or other category)
                Columns will have species
        """
        mtxs = {}
        dim1 = self.species_dim
        for analysis_code in self.other_dims:
            dim0 = analysis_code
            table_type = self.get_table_type(AGGREGATION_TYPE.MATRIX, dim0, dim1)

            # Dimension/Axis 0/row is always region or other analysis dimension
            meta = {
                "code": table_type,
                "fname": f"{table_type}{self.sep}{self.dt_token}",
                "file_extension": ".npz",
                "data_type": "matrix",

                # Dimensions: 0 is row (aka Axis 0), 1 is column (aka Axis 1)
                "dim_0_code": dim0,
                "dim_1_code": dim1,

                # These are all filled in for compressing data, reading data
                "row_categories": [],
                "column_categories": [],
                "value_fld": "",
                "datestr": self.dt_token,

                # Matrix values
                "value": SUMMARY_FIELDS.OCCURRENCE_COUNT,
            }
            mtxs[table_type] = meta
        return mtxs

    # ...........................
    def statistics(self):
        """Species by <dimension> statistics matrix/table defined for this project.

        Returns:
            stats (dict): dict of dictionaries for each matrix/table defined for this
                project.

        Note:
            Rows will always have analysis dimension (i.e. region or other category)
            Columns will have species
        """
        stats = {}
        # Axis 1 of PAM is always species
        dim1 = self.species_dim
        for analysis_code in self.other_dims:
            # Axis 0 of PAM is always 'site'
            dim0 = analysis_code
            table_type = self.get_table_type(AGGREGATION_TYPE.STATISTICS, dim0, dim1)
            meta = {
                "code": table_type,
                "fname": f"{table_type}{self.sep}{self.dt_token}",
                "file_extension": ".csv",
                "data_type": "stats",
                "datestr": self.dt_token,

                # Dimensions refer to the PAM matrix, site x species, from which the
                # stats are computed.
                "dim_0_code": dim0,
                "dim_1_code": dim1,

                # Minimum count defining 'presence' in the PAM
                "min_presence_count": 1,

                # TODO: Remove.  pandas.DataFrame contains row and column headers
                # # Categories refer to the statistics matrix headers
                # "row_categories": [],
                # "column_categories": []
            }
            stats[table_type] = meta
        return stats

    # ...........................
    def pam(self):
        """Species by <dimension> matrix defined for this project.

        Returns:
            pams (dict): dict of dictionaries for each matrix/table defined for this
                project.

        Note:
            Rows will always have analysis dimension (i.e. region or other category)
            Columns will have species
        """
        # TODO: Is this an ephemeral data structure used only for computing stats?
        #       If we want to save it, we must add compress_to_file,
        #       uncompress_zipped_data, read_data.
        #       If we only save computations, must save input HeatmapMatrix metadata
        #       and min_presence_count
        #       Note bison.spanalyst.pam_matrix.PAM
        pams = {}
        for analysis_code in self.other_dims:
            dim0 = analysis_code
            dim1 = self.species_dim
            table_type = self.get_table_type(AGGREGATION_TYPE.PAM, dim0, dim1)

            # Dimension/Axis 0/row is always region or other analysis dimension
            meta = {
                "code": table_type,
                "fname": f"{table_type}{self.sep}{self.dt_token}",
                "file_extension": ".npz",
                "data_type": "matrix",

                # Dimensions: 0 is row (aka Axis 0), 1 is column (aka Axis 1)
                "dim_0_code": dim0,
                "dim_1_code": dim1,

                # These are all filled in for compressing data, reading data
                "row_categories": [],
                "column_categories": [],
                "value_fld": "",
                "datestr": self.dt_token,

                # Matrix values
                "value": "presence",
                "min_presence_count": 1,
            }
            pams[table_type] = meta
        return pams

    # ...............................................
    def tables(self, datestr=None):
        """All tables of species count and occurrence count, summary, and matrix.

        Args:
            datestr (str): String in the format YYYY_MM_DD.

        Returns:
            sums (dict): dict of dictionaries for each table defined by the project.
                If datestr is provided, the token in the filename is replaced with that.

        Note:
            The keys for the dictionary (and code in the metadata values) are table_type
        """
        tables = self.list()
        tables.update(self.counts())
        tables.update(self.summary())
        tables.update(self.matrix())
        tables.update(self.pam())
        tables.update(self.statistics())
        if datestr is not None:
            # Update filename in summary tables
            for key, meta in tables.items():
                meta_cpy = deepcopy(meta)
                fname_tmpl = meta["fname"]
                meta_cpy["fname"] = fname_tmpl.replace(self.dt_token, datestr)
                tables[key] = meta_cpy
        return tables

    # ...............................................
    def get_table(self, table_type, datestr=None):
        """Update the filename in a metadata dictionary for one table, and return.

        Args:
            table_type: type of summary table to return.
            datestr: Datestring contained in the filename indicating the current version
                of the data.

        Returns:
            tables: dictionary of summary table metadata.

        Raises:
            Exception: on invalid table_type
        """
        tables = self.tables()
        try:
            table = tables[table_type]
        except KeyError:
            raise Exception(f"Invalid table_type {table_type}")
        cpy_table = deepcopy(table)
        if datestr is not None:
            cpy_table["datestr"] = datestr
            fname_tmpl = cpy_table["fname"]
            cpy_table["fname"] = fname_tmpl.replace(self.dt_token, datestr)
        return cpy_table

    # ...............................................
    def get_tabletype_from_filename_prefix(self, datacontents, datatype):
        """Get the table type from the file prefixes.

        Args:
            datacontents (str): first part of filename indicating data in table.
            datatype (str): second part of filename indicating form of data in table
                (records, list, matrix, etc).

        Returns:
            table_type (SUMMARY_TABLE_TYPES type): type of table.

        Raises:
            Exception: on invalid file prefix.
        """
        tables = self.tables()
        table_type = None
        for key, meta in tables.items():
            fname = meta["fname"]
            contents, dtp, _, _ = self._parse_filename(fname)
            if datacontents == contents and datatype == dtp:
                table_type = key
        if table_type is None:
            raise Exception(
                f"Table with filename prefix {datacontents}_{datatype} does not exist")
        return table_type

    # ...............................................
    def get_filename(self, table_type, datestr, is_compressed=False):
        """Update the filename in a metadata dictionary for one table, and return.

        Args:
            table_type (str): predefined type of data indicating type and contents.
            datestr (str): Datestring contained in the filename indicating the current version
                of the data.
            is_compressed (bool): flag indicating to return a filename for a compressed
                file.

        Returns:
            tables: dictionary of summary table metadata.
        """
        tables = self.tables()
        table = tables[table_type]
        ext = table["file_extension"]
        if is_compressed is True:
            ext = ZIP_EXTENSION
        fname_tmpl = f"{table['fname']}{ext}"
        fname = fname_tmpl.replace(self.dt_token, datestr)
        return fname

    # ...............................................
    def parse_table_type(self, table_type):
        """Parse the table_type into datacontents (dim0, dim1) and datatype.

        Args:
            table_type: String identifying the type of data and dimensions.

        Returns:
            datacontents (str): type of data contents
            dim0 (str): first dimension (rows/axis 0) of data in the table
            dim1 (str): second dimension (columns/axis 1) of data in the table
            datatype (str): type of data structure: summary table, stacked records
                (list or count), or matrix.

        Raises:
            Exception: on failure to parse table_type into 2 strings.
        """
        dim0 = dim1 = None
        fn_parts = table_type.split(self.sep)
        if len(fn_parts) >= 2:
            datatype = fn_parts.pop()
            idx = len(datatype) + 1
            datacontents = table_type[:-idx]
        else:
            raise Exception(f"Failed to parse {table_type}.")
        # Some data has 2 dimensions
        dim_parts = datacontents.split(self.dim_sep)
        dim0 = dim_parts[0]
        try:
            dim1 = dim_parts[1]
        except IndexError:
            pass
        return datacontents, dim0, dim1, datatype

    # ...............................................
    def _parse_filename(self, filename):
        # This will parse a filename for the compressed file of statistics, but
        #             not individual matrix and metadata files for each stat.
        # <datacontents>_<datatype>_<YYYY_MM_DD><_optional parquet extension>
        fname = os.path.basename(filename)
        if fname.endswith(S3_RS_TABLE_SUFFIX):
            stripped_fn = fname[:-len(S3_RS_TABLE_SUFFIX)]
            rest = S3_RS_TABLE_SUFFIX
        else:
            stripped_fn, ext = os.path.splitext(fname)
            rest = ext
        idx = len(stripped_fn) - len(self.dt_token)
        datestr = stripped_fn[idx:]
        table_type = stripped_fn[:idx-1]
        datacontents, dim0, dim1, datatype = self.parse_table_type(table_type)

        return datacontents, dim0, dim1, datatype, datestr, rest

    # ...............................................
    def get_tabletype_datestring_from_filename(self, filename):
        """Get the table type from the filename.

        Args:
            filename: relative or absolute filename of a SUMMARY data file.

        Returns:
            table_type (SUMMARY_TABLE_TYPES type): type of table.
            datestr (str): date of data in "YYYY_MM_DD" format.

        Raises:
            Exception: on failure to get tabletype and datestring from this filename.

        Note:
            This will parse a filename for the compressed file of statistics, but
            not individual matrix and metadata files for each stat.
        """
        try:
            datacontents, dim0, dim1, datatype, datestr, _rest = \
                self._parse_filename(filename)
            table_type = f"{datacontents}{self.sep}{datatype}"
        except Exception:
            raise
        return table_type, datestr


# .............................................................................
# Specify Network Workflow tasks: scripts, docker compose files, EC2 launch template versions,
#   Cloudwatch streams
class TASK:
    """Workflow tasks to be executed on EC2 instances."""
    userdata_extension = ".userdata.sh"

    # ...........................
    @classmethod
    def get_userdata_filename(cls, task, pth=None):
        """Get the filename containing userdata to execute this task.

        Args:
            task (str): task
            pth (str): local path for file.

        Returns:
            fname (str): filename for EC2 userdata to execute task.
        """
        fname = f"{task}{cls.userdata_extension}"
        if pth is not None:
            fname = os.path.join(pth, fname)
        return fname

    # ...........................
    @classmethod
    def get_task_from_userdata_filename(cls, fname):
        """Get the task for the userdata file.

        Args:
            fname (str): filename for EC2 userdata to execute task.

        Returns:
            task (str): task
        """
        task = fname.rstrip(cls.userdata_extension)
        return task
