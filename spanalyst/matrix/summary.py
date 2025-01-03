"""Matrix to summarize each of 2 dimensions of data by counts of the other and a third."""
from collections import OrderedDict
from copy import deepcopy
import pandas as pd

from spanalyst.common.constants import (
    AGGREGATION_TYPE, ANALYSIS_DIM, CSV_DELIMITER, SNKeys, SUMMARY, SUMMARY_FIELDS,
    TMP_PATH
)
from spanalyst.matrix.species_data import _SpeciesDataMatrix


# .............................................................................
class SummaryMatrix(_SpeciesDataMatrix):
    """Class for holding summary counts of each of 2 dimensions of data."""

    # ...........................
    def __init__(
            self, summary_df, table_type, datestr, dim0, dim1):
        """Constructor for occurrence/species counts by region/analysis_dim comparisons.

        Args:
            summary_df (pandas.DataFrame): DataFrame with a row for each element in
                category, and 2 columns of data.  Rows headers and column headers are
                labeled.
                * Column 1 contains the count of the number of columns in  that row
                * Column 2 contains the total of values in that row.
            table_type (aws_constants.SUMMARY_TABLE_TYPES): type of aggregated data
            datestr (str): date of the source data in YYYY_MM_DD format.
            dim0 (specnet.common.constants.ANALYSIS_DIM): dimension for axis 0, rows for
                which we will count and total dimension 1
            dim1 (specnet.common.constants.ANALYSIS_DIM): dimension for axis 1, with two
                columns (count and total) for each value in dimension 0.

        Note: Count and total dim1 for every value in dim0

        Note: constructed from records in table with datatype "counts" in
            spanalyst.common.constants.AGGREGATION_TYPE,
            i.e. county_x_riis_counts where each record has
                county, riis_status, occ_count, species_count;
                counts of occurrences and species by riis_status for a county
            OR
            county_counts where each record has
                county, occ_count, species_count;
                counts of occurrences and species for a county
        """
        self._df = summary_df

        _SpeciesDataMatrix.__init__(self, dim0, dim1, table_type, datestr)

    # ...........................
    @classmethod
    def init_from_heatmap(cls, heatmap, axis=0):
        """Summarize a matrix into counts of one axis and values for the other axis.

        Args:
            heatmap (spanalyst.matrix.heatmap.HeatmapMatrix): A 2d sparse matrix with
                count values for one dimension, (i.e. region) rows (axis 0), by the
                species dimension, columns (axis 1), to use for computations.
            axis (int): Summarize rows (0) for each column, or columns (1) for each row.

        Returns:
            sparse_coo (pandas.DataFrame): DataFrame summarizing rows by the count and
                value of columns, or columns by the count amd value of rows.

        Note:
            The input dataframe must contain only one input record for any x and y value
                combination, and each record must contain another value for the
                dataframe contents.

        Note:
            Total/Count down axis 0/row, rows for a column
                across axis 1/column, columns for a row (aka species)
            Axis 0 produces a matrix shape (col_count, 1),
                1 row of values, total for each column
            Axis 1 produces matrix shape (row_count, 1),
                1 column of values, total for each row
        """
        # Total of the values along the axis
        totals = heatmap.get_totals(axis=axis)
        # Count of the non-zero values along the axis
        counts = heatmap.get_counts(axis=axis)
        data = {SUMMARY_FIELDS.COUNT: counts, SUMMARY_FIELDS.TOTAL: totals}

        # Sparse matrix always has species in axis 1.
        species_dim = heatmap.x_dimension
        other_dim = heatmap.y_dimension

        # Axis 0 summarizes down axis 0, each column/species, other dimension
        # (i.e. region) counts and occurrence totals of other dimension in sparse matrix
        if axis == 0:
            dim0 = species_dim
            dim1 = other_dim
            index = heatmap.column_category.categories
            table_type = SUMMARY.get_table_type(
                AGGREGATION_TYPE.SUMMARY, species_dim["code"], other_dim["code"])
        # Axis 1 summarizes across axis 1, each row/other dimension, species counts and
        # occurrence totals of species in sparse matrix
        elif axis == 1:
            dim0 = other_dim
            dim1 = species_dim
            index = heatmap.row_category.categories
            table_type = SUMMARY.get_table_type(
                AGGREGATION_TYPE.SUMMARY, other_dim["code"], species_dim["code"])

        # summary fields = columns, sparse matrix axis = rows
        sdf = pd.DataFrame(data=data, index=index)

        summary_matrix = SummaryMatrix(sdf, table_type, heatmap.datestr, dim0, dim1)
        return summary_matrix

    # ...........................
    @classmethod
    def init_from_compressed_file(
            cls, zip_filename, local_path=TMP_PATH, overwrite=False):
        """Construct a SparseMatrix from a compressed file.

        Args:
            zip_filename (str): Filename of zipped sparse matrix data to uncompress.
            local_path (str): Absolute path of local destination path
            overwrite (bool): Flag indicating whether to use existing files unzipped
                from the zip_filename.

        Returns:
            sparse_mtx (spanalyst.matrix.sparse_matrix.SparseMatrix): data matrix.

        Raises:
            Exception: on failure to uncompress files.
            Exception: on failure to load data from uncompressed files.

        Note:
            All filenames have the same basename with extensions indicating which data
                they contain. The filename contains a string like YYYY-MM-DD which
                indicates which GBIF data dump the statistics were built upon.
        """
        try:
            dataframe, meta_dict, table_type, datestr = cls.uncompress_zipped_data(
                zip_filename, local_path=local_path, overwrite=overwrite)
        except Exception:
            raise

        dim0 = ANALYSIS_DIM.get(meta_dict["dim_0_code"])
        dim1 = ANALYSIS_DIM.get(meta_dict["dim_1_code"])
        # Create
        summary_mtx = SummaryMatrix(dataframe, table_type, datestr, dim0, dim1)

        return summary_mtx

    # ...............................................
    @property
    def num_items(self):
        """Get the number of rows (each with measurements).

        Returns:
            int: The count of rows
        """
        return self._df.shape[0]

    # ...............................................
    @property
    def num_measures(self):
        """Get the number of columns (measurements).

        Returns:
            int: The count of columns
        """
        return self._df.shape[1]

    # ...............................................
    def get_random_row_labels(self, count):
        """Get random values from the labels on axis 0 of matrix.

        Args:
            count (int): number of values to return

        Returns:
            labels (list): random row headers
        """
        import random
        size = len(self._df.index)
        # Get a random sample of category indexes (0-based)
        idxs = random.sample(range(size), count)
        labels = [self._df.index[i] for i in idxs]
        return labels

    # .............................................................................
    def compress_to_file(self, local_path=TMP_PATH):
        """Compress this SparseMatrix to a zipped npz and json file.

        Args:
            local_path (str): Absolute path of local destination path

        Returns:
            zip_fname (str): Local output zip filename.

        Raises:
            Exception: on failure to write dataframe to CSV file.
            Exception: on failure to serialize or write metadata as JSON.
            Exception: on failure to write matrix and metadata files to zipfile.
        """
        # Always delete local files before compressing this data.
        [mtx_fname, meta_fname, zip_fname] = self._remove_expected_files(
            local_path=local_path)

        # Save matrix to csv locally
        try:
            self._df.to_csv(mtx_fname, sep=CSV_DELIMITER)
        except Exception as e:
            msg = f"Failed to write {mtx_fname}: {e}"
            raise Exception(msg)

        # Save table data and categories to json locally
        metadata = deepcopy(self._table)
        # Should be filled already, make sure they are consistent!
        if metadata["dim_0_code"] != self.y_dimension["code"]:
            raise Exception(
                f"metadata/dim_0_code {metadata['dim_0_code']} != "
                f"y_dimension {self.y_dimension['code']}"
            )
        if metadata["dim_1_code"] != self.x_dimension["code"]:
            raise Exception(
                f"metadata/dim_1_code {metadata['dim_1_code']} != "
                f"x_dimension {self.x_dimension['code']}"
            )
        try:
            self._dump_metadata(metadata, meta_fname)
        except Exception:
            raise

        # Compress matrix with metadata
        try:
            self._compress_files([mtx_fname, meta_fname], zip_fname)
        except Exception:
            raise

        return zip_fname

    # .............................................................................
    @classmethod
    def uncompress_zipped_data(
            cls, zip_filename, local_path=TMP_PATH, overwrite=False):
        """Uncompress a zipped SparseMatrix into a coo_array and row/column categories.

        Args:
            zip_filename (str): Filename of output data to write to S3.
            local_path (str): Absolute path of local destination path
            overwrite (bool): Flag indicating whether to use existing files unzipped
                from the zip_filename.

        Returns:
            dataframe (pandas.DataFrame): dataframe containing summary matrix data.
            meta_dict (dict): metadata for the matrix
            table_type (aws.aws_constants.SUMMARY_TABLE_TYPES): type of table data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on failure to uncompress files.
            Exception: on failure to load data from uncompressed files.
        """
        try:
            mtx_fname, meta_fname, table_type, datestr = cls._uncompress_files(
                zip_filename, local_path, overwrite=overwrite)
        except Exception:
            raise

        # Read matrix data from local files
        try:
            dataframe, meta_dict = cls.read_data(mtx_fname, meta_fname)
        except Exception:
            raise

        return dataframe, meta_dict, table_type, datestr

    # .............................................................................
    @classmethod
    def read_data(cls, mtx_filename, meta_filename):
        """Read SummaryMatrix data files into a dataframe and metadata dictionary.

        Args:
            mtx_filename (str): Filename of pandas.DataFrame data in csv format.
            meta_filename (str): Filename of JSON summary matrix metadata.

        Returns:
            dataframe (pandas.DataFrame): dataframe containing summary matrix data.
            meta_dict (dict): metadata for the matrix
            table_type (aws.aws_constants.SUMMARY_TABLE_TYPES): type of table data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on unable to load CSV file
            Exception: on unable to load JSON metadata
        """
        # Read dataframe from local CSV file
        try:
            dataframe = pd.read_csv(mtx_filename, sep=CSV_DELIMITER, index_col=0)
        except Exception as e:
            raise Exception(f"Failed to load {mtx_filename}: {e}")
        # Read JSON dictionary as string
        try:
            meta_dict = cls.load_metadata(meta_filename)
        except Exception:
            raise

        return dataframe, meta_dict

    # ...............................................
    def get_row_values(self, row_label):
        """Get the labels and values for a row.

        Args:
            row_label: label of row to get values for.

        Returns:
            row_dict: Dictionary of labels and values for the row.

        Raises:
            Exception: on failure to find row_label in the dataframe index.
        """
        vals = {}
        try:
            row = self._df.loc[[row_label]].to_dict()
        except KeyError:
            raise Exception(f"Failed to find row {row_label} in index")

        for lbl, valdict in row.items():
            vals[lbl] = valdict[row_label]
        return vals

    # ...............................................
    def get_measures(self, summary_key):
        """Get a dictionary of statistics for the summary row with this label.

        Args:
            summary_key: label on the row to gather stats for.

        Returns:
            stats (dict): quantitative measures of the item.
        """
        # Get measurements (pandas series)
        measures = self._df.loc[summary_key]
        stats = {
            self._keys[SNKeys.ONE_LABEL]: summary_key,
            self._keys[SNKeys.ONE_COUNT]: measures.loc[SUMMARY_FIELDS.COUNT],
            self._keys[SNKeys.ONE_TOTAL]: measures.loc[SUMMARY_FIELDS.TOTAL]
        }
        return stats

    # ...........................
    def rank_measures(self, sort_by, order="descending", limit=10):
        """Order records by sort_by field and return the top or bottom limit records.

        Args:
            sort_by (str): field containing measurement to sort on
                (options: SUMMARY_FIELDS.COUNT, SUMMARY_FIELDS.TOTAL).
            order (str): return records, sorted from top (descending) or bottom
                (ascending).
            limit (int): number of records to return.

        Returns:
            ordered_rec_dict (OrderedDict): records containing all fields, sorted by the
                sort_by field.

        Raises:
            Exception: on sort field does not exist in data.
        """
        measure_flds = self._keys["fields"].copy()
        try:
            measure_flds.remove(sort_by)
        except ValueError:
            raise Exception(
                f"Field {sort_by} does not exist; sort by one of {self._keys['fields']}")
        # Get largest and down
        if order == "descending":
            sorted_df = self._df.nlargest(limit, sort_by, keep="all")
        # Get smallest and up
        elif order == "ascending":
            sorted_df = self._df.nsmallest(limit, sort_by, keep="all")
        else:
            raise Exception(
                f"Order {sort_by} does not exist, use 'ascending' or 'descending')")
        # Returns dict with each measurement in a separate dictionary, so re-arrange
        rec_dict = sorted_df.to_dict()
        ordered_rec_dict = OrderedDict()
        # Create records from the sorted measurement first, in order returned
        for k, v in rec_dict[sort_by]:
            ordered_rec_dict[k] = {sort_by: v}
            # measure_flds now contains all fields except sort_by
            #   add remaining fields and values to the new ordered records
            for other_fld in measure_flds:
                ordered_rec_dict[k][other_fld] = rec_dict[other_fld][k]
        return ordered_rec_dict
