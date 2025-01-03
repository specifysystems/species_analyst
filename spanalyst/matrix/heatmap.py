"""Matrix to summarize 2 dimensions of data by counts of a third in a sparse matrix."""
from copy import deepcopy
from logging import ERROR
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import random
import scipy.sparse

from spanalyst.common.constants import ANALYSIS_DIM, SNKeys, TMP_PATH
from spanalyst.matrix.species_data import _SpeciesDataMatrix


# .............................................................................
class HeatmapMatrix(_SpeciesDataMatrix):
    """Class for managing computations for counts of aggregator0 x aggregator1."""

    # ...........................
    def __init__(
            self, sparse_coo_array, table_type, datestr, row_category,
            column_category, dim0, dim1, val_fld):
        """Constructor for species by region/analysis_dim comparisons.

        Args:
            sparse_coo_array (scipy.sparse.coo_array): A 2d sparse array with count
                values for one dimension (i.e. region) rows (axis 0) by the
                species dimension columns (axis 1) to use for computations.
            table_type (specnet.tools.s2n.constants.SUMMARY_TABLE_TYPES): type of
                aggregated data
            datestr (str): date of the source data in YYYY_MM_DD format.
            row_category (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows.
            column_category (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns.
            dim0 (specnet.common.constants.ANALYSIS_DIM): dimension for axis 0, rows
            dim1 (specnet.common.constants.ANALYSIS_DIM.SPECIES): dimension for axis 1,
                columns, always species dimension in specnet sparse matrices.
            val_fld (str): column header from stacked input records containing values
                for sparse matrix cells

        Raises:
            Exception: on input matrix not of correct type.
            Exception: on provided dimension 0 differs from table_type dimension 0.
            Exception: on provided dimension 1 differs from table_type dimension 1.
            Exception: on provided dimension 1 is not species dimension

        Note: the current Specnet implementation of HeatmapMatrix expects the species
            dimension to be columns (axis/dimension 1).
            Checks for this:
                this constructor raises an Exception
                specnet.common.constants.SUMMARY matrix table_type defines
                    dim1 = spanalyst.common.constants.SPECIES

        Note: y_fld, x_fld, val_fld refer to column headers from the original data
            used to construct the sparse matrix.  They are included to aid in testing
            the original data against the sparse matrix.

        Note: constructed from `stacked` records in table with datatype "list" in
            specnet.common.constants.SUMMARY.DATATYPES, i.e. county_x_species_list
            where each record has county, species, riis_status, occ_count, a list of
            species in a county.

        Note: this matrix approximates a heatmap, and can easily be converted to a
            Presence Absence Matrix (PAM), sites by species, with sites ~= region
            (or other dimension that applies to all records for comparison).
        """
        if type(sparse_coo_array) != scipy.sparse.coo_array:
            raise Exception("Input matrix must be in scipy.sparse.coo_array format")
        self._coo_array = sparse_coo_array
        self._row_categ = row_category
        self._col_categ = column_category
        self._val_fld = val_fld
        if dim1["code"] != ANALYSIS_DIM.species_code():
            raise Exception("HeatmapMatrix requires dimension1 (columns) to be Species")

        _SpeciesDataMatrix.__init__(self, dim0, dim1, table_type, datestr)

    # ...........................
    @property
    def shape(self):
        """Return analysis dimension for axis 1.

        Returns:
            (specnet.common.constants.ANALYSIS_DIM): Data dimension for axis 1 (columns).
        """
        return self._coo_array.shape

    # ...........................
    @property
    def sparse_array(self):
        """Return sparse_array.

        Returns:
            (scipy.sparse.coo_array): Sparse_array for the object.
        """
        return self._coo_array

    # ...........................
    @property
    def data(self):
        """Return sparse_array data.

        Returns:
            (scipy.sparse.coo_array): Data in sparse_array.
        """
        return self._coo_array.data

    # ...........................
    @classmethod
    def init_from_compressed_file(
            cls, zip_filename, local_path=TMP_PATH, overwrite=False):
        """Construct a HeatmapMatrix from a compressed file.

        Args:
            zip_filename (str): Filename of zipped sparse matrix data to uncompress.
            local_path (str): Absolute path of local destination path
            overwrite (bool): Flag indicating whether to use existing files unzipped
                from the zip_filename.

        Returns:
            heatmap (spanalyst.matrix.heatmap_matrix.HeatmapMatrix): matrix for the data.

        Raises:
            Exception: on failure to uncompress files.
            Exception: on failure to load data from uncompressed files.

        Note:
            All filenames have the same basename with extensions indicating which data
                they contain. The filename contains a string like YYYY-MM-DD which
                indicates which GBIF data dump the statistics were built upon.
        """
        try:
            mtx_fname, meta_fname, table_type, datestr = cls._uncompress_files(
                zip_filename, local_path=local_path, overwrite=overwrite)
        except Exception:
            raise

        try:
            sparse_coo, meta_dict, row_categ, col_categ = cls.read_data(
                mtx_fname, meta_fname)
        except Exception:
            raise

        dim0 = ANALYSIS_DIM.get(meta_dict["dim_0_code"])
        dim1 = ANALYSIS_DIM.get(meta_dict["dim_1_code"])

        # Create
        heatmap = HeatmapMatrix(
            sparse_coo, meta_dict["code"], datestr, row_categ, col_categ, dim0, dim1,
            meta_dict["value_fld"])

        return heatmap

    # ...........................
    @classmethod
    def init_from_stacked_data(
            cls, stacked_df, y_fld, x_fld, val_fld, table_type, datestr):
        """Create a sparse matrix of rows by columns containing values from a table.

        Args:
            stacked_df (pandas.DataFrame): DataFrame of records containing columns to be
                used as the new rows, new columns, and values.
            y_fld: column in the input dataframe containing values to be used as rows
                (axis 0)
            x_fld: column in the input dataframe containing values to be used as
                columns (axis 1)
            val_fld: : column in the input dataframe containing values to be used as
                values for the intersection of x and y fields
            table_type (specnet.tools.s2n.constants.SUMMARY_TABLE_TYPES): table type of
                sparse matrix aggregated data
            datestr (str): date of the source data in YYYY_MM_DD format.

        Returns:
            sparse_matrix (spanalyst.matrix.heatmap_matrix.HeatmapMatrix): matrix of y values
                (rows, y axis=0) by x values (columnns, x axis=1), with values from
                another column.

        Raises:
            Exception: on failure to find a dimension for the fields to be used for the
                x and y axes.

        Note:
            The input dataframe must contain only one input record for any x and y value
                combination, and each record must contain another value for the dataframe
                contents.  The function was written for a table of records with
                datasetkey (for the column labels/x), species (for the row labels/y),
                and occurrence count.
        """
        # Check that x,y fields correspond to a known data dimension
        try:
            _ = ANALYSIS_DIM.get_from_key_fld(y_fld)
            _ = ANALYSIS_DIM.get_from_key_fld(x_fld)
        except Exception:
            raise

        # Get unique values to use as categories for scipy column and row indexes,
        # remove None
        unique_x_vals = list(stacked_df[x_fld].dropna().unique())
        unique_y_vals = list(stacked_df[y_fld].dropna().unique())
        # Categories allow using codes as the integer index for scipy matrix
        row_categ = CategoricalDtype(unique_y_vals, ordered=True)
        col_categ = CategoricalDtype(unique_x_vals, ordered=True)
        # Create a list of category codes matching original stacked data to replace
        #   column names from stacked data dataframe with integer codes for row and
        #   column indexes in the new scipy matrix
        col_idx = stacked_df[x_fld].astype(col_categ).cat.codes
        row_idx = stacked_df[y_fld].astype(row_categ).cat.codes

        dim0 = ANALYSIS_DIM.get_from_key_fld(y_fld)
        dim1 = ANALYSIS_DIM.get_from_key_fld(x_fld)
        # This creates a new matrix in Coordinate list (COO) format.  COO stores a list
        # of (row, column, value) tuples.  Convert to CSR or CSC for efficient Row or
        # Column slicing, respectively
        sparse_coo = scipy.sparse.coo_array(
            (stacked_df[val_fld], (row_idx, col_idx)),
            shape=(row_categ.categories.size, col_categ.categories.size))
        sparse_matrix = HeatmapMatrix(
            sparse_coo, table_type, datestr, row_categ, col_categ,
            dim0, dim1, val_fld=val_fld)
        return sparse_matrix

    # ...........................
    @property
    def input_val_fld(self):
        """Return column header from input data records (stacked data) for matrix value.

        Returns:
            (str): Input field name for values (matrix cells).
        """
        return self._val_fld

    # ...........................
    @property
    def row_category(self):
        """Return the data structure representing the row category.

        Returns:
            self._row_categ (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows.
        """
        return self._row_categ

    # ...........................
    @property
    def column_category(self):
        """Return the data structure representing the column category.

        Returns:
            self._col_categ (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns.
        """
        return self._col_categ

    # .............................................................................
    def _to_dataframe(self):
        sdf = pd.DataFrame.sparse.from_spmatrix(
            self._coo_array,
            index=self._row_categ.categories,
            columns=self._col_categ.categories)
        return sdf

    # # ...............................................
    # def _get_code_from_category(self, label, axis=0):
    #     if axis == 0:
    #         categ = self._row_categ
    #     elif axis == 1:
    #         categ = self._col_categ
    #     else:
    #         raise Exception(f"2D sparse array does not have axis {axis}")
    #
    #     # returns a tuple of a single 1-dimensional array of locations
    #     arr = np.where(categ.categories == label)[0]
    #     try:
    #         # labels are unique in categories so there will be 0 or 1 value in the array
    #         code = arr[0]
    #     except IndexError:
    #         raise
    #     return code
    #
    # # ...............................................
    # def _get_category_from_code(self, code, axis=0):
    #     if axis == 0:
    #         categ = self._row_categ
    #     elif axis == 1:
    #         categ = self._col_categ
    #     else:
    #         raise Exception(f"2D sparse array does not have axis {axis}")
    #     category = categ.categories[code]
    #     return category
    #
    # # ...............................................
    # def _get_categories_from_code(self, code_list, axis=0):
    #     if axis == 0:
    #         categ = self._row_categ
    #     elif axis == 1:
    #         categ = self._col_categ
    #     else:
    #         raise Exception(f"2D sparse array does not have axis {axis}")
    #     category_labels = []
    #     for code in code_list:
    #         category_labels.append(categ.categories[code])
    #     return category_labels

    # ...............................................
    def _export_categories(self, axis=0):
        if axis == 0:
            categ = self._row_categ
        elif axis == 1:
            categ = self._col_categ
        else:
            raise Exception(f"2D sparse array does not have axis {axis}")
        cat_lst = categ.categories.tolist()
        return cat_lst

    # ...............................................
    @classmethod
    def _get_code_from_category(cls, label, categ):
        # returns a tuple of a single 1-dimensional array of locations
        arr = np.where(categ.categories == label)[0]
        try:
            # labels are unique in categories so there will be 0 or 1 value in the array
            code = arr[0]
        except IndexError:
            raise
        return code

    # ...............................................
    @classmethod
    def _get_category_from_code(self, code, categ):
        category = categ.categories[code]
        return category

    # ...............................................
    @classmethod
    def _get_categories_from_code(self, code_list, categ):
        category_labels = []
        for code in code_list:
            category_labels.append(categ.categories[code])
        return category_labels

    # ...........................
    def _to_csr(self):
        # Convert to CSR format for efficient row slicing
        csr = self._coo_array.tocsr()
        return csr

    # ...........................
    def _to_csc(self):
        # Convert to CSC format for efficient column slicing
        csc = self._coo_array.tocsr()
        return csc

    # ...........................
    @classmethod
    def _get_nonzero_labels_zero_indexes(
            cls, nonzero_ridx, nonzero_cidx, row_categ, col_categ):
        # Save indexes of all-zero columns,
        zero_cidx = []
        zero_ridx = []
        # Save category/labels of columns with at least one non-zero
        nonzero_col_labels = []
        nonzero_row_labels = []

        # Create categories for only the rows, columns that are not all zero
        for axis, nonzero_idx, categ, zero_idx, nonzero_labels in (
                (0, nonzero_ridx, row_categ, zero_ridx, nonzero_row_labels),
                (1, nonzero_cidx, col_categ, zero_cidx, nonzero_col_labels)
        ):
            # Examine each non-zero position found
            for zidx in range(len(nonzero_idx)):
                # If true that this position contains a non-zero in the row/column
                #   value is type numpy.bool_ and `is True` only works after typecasting
                if bool(nonzero_idx[zidx]) is True:
                    # Save labels with non-zero elements for new category index
                    label = cls._get_category_from_code(zidx, categ)
                    nonzero_labels.append(label)
                else:
                    # Save position with only zero elements for deletion
                    zero_idx.append(zidx)

            # Compile categories from nonzero labels, assign to row or column var
            if axis == 0:
                cmp_row_categ = CategoricalDtype(nonzero_labels, ordered=True)
            else:
                cmp_col_categ = CategoricalDtype(nonzero_labels, ordered=True)

        return (zero_ridx, zero_cidx, cmp_row_categ, cmp_col_categ)

    # ...........................
    @classmethod
    def _remove_zeros(cls, coo, row_categ, col_categ):
        """Remove any all-zero rows or columns.

        Args:
            coo (scipy.sparse.coo_array): binary sparse array in coo format
            row_categ (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows in the input matrix.
            col_categ (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns in the input matrix.

        Returns:
            compressed_coo (scipy.sparse.coo_array): sparse array with no rows or
                columns containing all zeros.
            row_category (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows in the new compressed matrix.
            column_category (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns in the new compressed matrix.
        """
        # Get indices of col/rows that contain at least one non-zero element, with dupes
        nz_cidx = scipy.sparse.find(coo)[1]
        nz_ridx = scipy.sparse.find(coo)[0]

        # Get a bool array with elements T if position holds a nonzero
        nonzero_cidx = np.isin(np.arange(coo.shape[1]), nz_cidx)
        nonzero_ridx = np.isin(np.arange(coo.shape[0]), nz_ridx)

        # WARNING: Indices of altered axes are reset in the returned matrix, so we
        #   will recreate categories with only non-zero vectors
        (zero_ridx, zero_cidx, cmp_row_categ, cmp_col_categ
         ) = cls._get_nonzero_labels_zero_indexes(
            nonzero_ridx, nonzero_cidx, row_categ, col_categ)

        # Construct masks
        csr = coo.tocsr()
        if len(zero_cidx) > 0:
            col_mask = np.ones(csr.shape[1], dtype=bool)
            col_mask[zero_cidx] = False
        if len(zero_ridx) > 0:
            row_mask = np.ones(csr.shape[0], dtype=bool)
            row_mask[zero_ridx] = False

        # Mask with indices to remove data
        if len(zero_ridx) > 0 and len(zero_cidx) > 0:
            compressed_csr = csr[row_mask][:, col_mask]
        elif len(zero_ridx) > 0:
            compressed_csr = csr[row_mask]
        elif len(zero_cidx) > 0:
            compressed_csr = csr[:, col_mask]
        else:
            compressed_csr = csr
        cmp_coo = compressed_csr.tocoo()

        return cmp_coo, cmp_row_categ, cmp_col_categ

    # ...............................................
    def filter(self, min_count=None, max_count=None):
        """Filter the coo_array by parameters, then remove all-zero rows and columns.

        Args:
            min_count (int): filter all values below this value.
            max_count (int): filter all values above this value.

        Returns:
            coo_array (scipy.sparse.coo_array): A 2d sparse array where values in the
                array meet all of the provided conditions.
            row_categ (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows in the new filtered matrix.
            column_categ (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns in the new filtered matrix.

        Raises:
            Exception: on any filter parameter (min_count, max_count, divisible_by) < 1
            Exception: on no parameters provided.
        """
        for pmt in (min_count, max_count):
            if pmt is not None and pmt <= 0:
                raise Exception(f"Filter parameter {pmt} must be an integer >= 1")
        if (min_count is None and max_count is None):
            raise Exception("No filters provided")

        csr_array = self._coo_array.tocsr()
        # Returns a CSR array of the same shape with filtered items set to zeros
        if min_count is not None:
            csr_array = csr_array.multiply(csr_array >= min_count)
        if max_count is not None:
            csr_array = csr_array.multiply(max_count >= csr_array)

        coo_array = csr_array.tocoo()

        cmp_coo, cmp_row_categ, cmp_col_categ = HeatmapMatrix._remove_zeros(
            coo_array, self._row_categ, self._col_categ)

        new_heatmap = HeatmapMatrix(
            cmp_coo, self._table_type, self._datestr, cmp_row_categ, cmp_col_categ,
            self._row_dim, self._col_dim, self._val_fld)

        return new_heatmap

    # ...............................................
    def get_random_labels(self, count, axis=0):
        """Get random values from the labels on an axis of a sparse matrix.

        Args:
            count (int): number of values to return
            axis (int): row (0) or column (1) header for labels to gather.

        Returns:
            x_vals (list): random values pulled from the column

        Raises:
            Exception: on axis not in (0, 1)
        """
        if axis == 0:
            categ = self._row_categ
        elif axis == 1:
            categ = self._col_categ
        else:
            raise Exception(f"2D sparse array does not have axis {axis}")
        # Get a random sample of category indexes
        idxs = random.sample(range(1, len(categ.categories)), count)
        labels = [self._get_category_from_code(i, categ) for i in idxs]
        return labels

    # ...............................................
    @property
    def num_y_values(self):
        """Get the number of rows.

        Returns:
            int: The count of rows where the value > 0 in at least one column.

        Note:
            Also used as gamma diversity (species richness over entire landscape)
        Note: because the sparse data will only from contain unique rows and columns
            with data, this should ALWAYS equal the number of rows
        """
        return self._coo_array.shape[0]

    # ...............................................
    @property
    def num_x_values(self):
        """Get the number of columns.

        Returns:
            int: The count of columns where the value > 0 in at least one row

        Note: because the sparse data will only from contain unique rows and columns
            with data, this should ALWAYS equal the number of columns
        """
        return self._coo_array.shape[1]

    # ...............................................
    def get_vector_from_label(self, label, axis=0):
        """Return the row (axis 0) or column (axis 1) with label `label`.

        Args:
            label: label for row of interest
            axis (int): row (0) or column (1) header for vector and index to gather.

        Returns:
            vector (scipy.sparse.csr_array): 1-d array of the row/column for 'label'.
            idx (int): index for the vector (zeros and non-zeros) in the sparse matrix

        Raises:
            Exception: on label does not exist in category
            Exception: on axis not in (0, 1)
        """
        if axis == 0:
            categ = self._row_categ
        elif axis == 1:
            categ = self._col_categ
        try:
            idx = self._get_code_from_category(label, categ)
        except IndexError:
            axis_type = self.y_dimension["code"]
            if axis == 1:
                axis_type = self.x_dimension["code"]
            raise Exception(f"Label {label} does not exist in axis {axis}, {axis_type}")
        lil = self._coo_array.tolil()
        if axis == 0:
            vector = lil.getrow(idx)
        elif axis == 1:
            vector = lil.T.getrow(idx)
        else:
            raise Exception(f"2D sparse array does not have axis {axis}")
        idx = self.convert_np_vals_for_json(idx)
        return vector, idx

    # ...............................................
    def sum_vector(self, label, axis=0):
        """Get the total of values in a single row or column.

        Args:
            label: label on the row (axis 0) or column (axis 1) to total.
            axis (int): row (0) or column (1) header for vector to sum.

        Returns:
            int: The total of all values in one column

        Raises:
            IndexError: on label not present in vector header
        """
        try:
            vector, _idx = self.get_vector_from_label(label, axis=axis)
        except IndexError:
            raise
        total = vector.sum()
        return total

    # ...............................................
    def count_vector(self, label, axis=0):
        """Count non-zero values in a single row or column.

        Args:
            label: label on the row (axis 0) or column (axis 1) to total.
            axis (int): row (0) or column (1) header for vector to sum.

        Returns:
            int: The count of all non-zero values in one column

        Raises:
            IndexError: on label not present in vector header
        """
        try:
            vector, _idx = self.get_vector_from_label(label, axis=axis)
        except IndexError:
            raise
        count = vector.getnnz()
        return count

    # ...............................................
    def sum_vector_ge_than(self, label, min_val, axis=0):
        """Get the total of values >= min_val in a single row or column.

        Args:
            label: label on the row (axis 0) or column (axis 1) to total.
            min_val (int): minimum value to be included in sum.
            axis (int): row (0) or column (1) header for vector to sum.

        Returns:
            int: The total of all values >= min_val in one column

        Raises:
            IndexError: on label not present in vector header
        """
        try:
            vector, _idx = self.get_vector_from_label(label, axis=axis)
        except IndexError:
            raise

        # Create a mask for values greater than x
        mask = vector >= min_val

        # Sum the values greater than x
        sum_ge_than = np.sum(vector[mask])
        return sum_ge_than

    # ...............................................
    def count_vector_ge_than(self, label, min_val, axis=0):
        """Get the count of values >= min_val in a single row or column.

        Args:
            label: label on the row (axis 0) or column (axis 1) to total.
            min_val (int): minimum value to be included in count.
            axis (int): row (0) or column (1) header for vector to sum.

        Returns:
            int: The count of all values >= min_val in one column

        Raises:
            IndexError: on label not present in vector header
        """
        try:
            vector, _idx = self.get_vector_from_label(label, axis=axis)
        except IndexError:
            raise

        # Create a mask for values greater than x
        mask = vector >= min_val

        # Sum the values greater than x
        count_ge_than = len(vector[mask])
        return count_ge_than

    # ...............................................
    def get_row_labels_for_data_in_column(self, col, value=None):
        """Get the minimum or maximum NON-ZERO value and row label(s) for a column.

        Args:
            col: column to find row labels in.
            value: filter data value to return row labels for.  If None, return labels
                for all non-zero rows.

        Returns:
            target: The minimum or maximum value for a column
            row_labels: The labels of the rows containing the target value
        """
        # Returns row_idxs, col_idxs, vals of NNZ values in row
        row_idxs, col_idxs, vals = scipy.sparse.find(col)
        if value is None:
            idxs_lst = [row_idxs[i] for i in range(len(row_idxs))]
        else:
            tmp_idxs = np.where(vals == value)[0]
            tmp_idx_lst = [tmp_idxs[i] for i in range(len(tmp_idxs))]
            # Row indexes of maxval in column
            idxs_lst = [row_idxs[i] for i in tmp_idx_lst]
        row_labels = [
            self._get_category_from_code(idx, self._row_categ) for idx in idxs_lst
        ]
        return row_labels

    # ...............................................
    def get_extreme_val_labels_for_vector(self, vector, axis=0, is_max=True):
        """Get the minimum or maximum NON-ZERO value and axis label(s) for a vecto.

        Args:
            vector (numpy.array): 1 dimensional array for a row or column.
            is_max (bool): flag indicating whether to get maximum (T) or minimum (F)
            axis (int): row (0) or column (1) header for extreme value and labels.

        Returns:
            target: The minimum or maximum value for a column
            row_labels: The labels of the rows containing the target value
        """
        # Returns row_idxs, col_idxs, vals of NNZ values in row
        row_idxs, col_idxs, vals = scipy.sparse.find(vector)
        if is_max is True:
            target = vals.max()
        else:
            target = vals.min()
        target = self.convert_np_vals_for_json(target)

        # Get labels for this value in
        labels = self.get_labels_for_val_in_vector(vector, target, axis=axis)
        return target, labels

    # ...............................................
    def get_labels_for_val_in_vector(self, vector, target_val, axis=0):
        """Get the row or column label(s) for a vector containing target_val.

        Args:
            vector (numpy.array): 1 dimensional array for a row or column.
            target_val (int): value to search for in a row or column
            axis (int): row (0) or column (1) header for extreme value and labels.

        Returns:
            target: The minimum or maximum value for a column
            row_labels: The labels of the rows containing the target value

        Raises:
            Exception: on axis not in (0, 1)
        """
        # Returns row_idxs, col_idxs, vals of NNZ values in row
        row_idxs, col_idxs, vals = scipy.sparse.find(vector)

        # Get indexes of target value within NNZ vals
        tmp_idxs = np.where(vals == target_val)[0]
        tmp_idx_lst = [tmp_idxs[i] for i in range(len(tmp_idxs))]
        # Get actual indexes (within all zero/non-zero elements) of target in vector
        if axis == 0:
            # Column indexes of maxval in row
            idxs_lst = [col_idxs[i] for i in tmp_idx_lst]
            # Label category is the opposite of the vector axis
            label_categ = self._col_categ
        elif axis == 1:
            # Row indexes of maxval in column
            idxs_lst = [row_idxs[j] for j in tmp_idx_lst]
            label_categ = self._row_categ
        else:
            raise Exception(f"2D sparse array does not have axis {axis}")

        # Convert from indexes to labels
        labels = [
            self._get_category_from_code(idx, label_categ) for idx in idxs_lst]
        return labels

    # ...............................................
    def count_val_in_vector(self, vector, target_val):
        """Count the row or columns containing target_val in a vector.

        Args:
            vector (numpy.array): 1 dimensional array for a row or column.
            target_val (int): value to search for in a row or column

        Returns:
            target: The minimum or maximum value for a column
            row_labels: The labels of the rows containing the target value
        """
        # Returns row_idxs, col_idxs, vals of NNZ values in row
        row_idxs, col_idxs, vals = scipy.sparse.find(vector)
        # Get indexes of target value within NNZ vals
        tmp_idxs = np.where(vals == target_val)[0]
        tmp_idx_lst = [tmp_idxs[i] for i in range(len(tmp_idxs))]
        count = len(tmp_idx_lst)
        return count

    # ...............................................
    def get_row_stats(self, row_label=None):
        """Get the statistics for one or all rows.

        Args:
            row_label (str): label for one row of data to examine.

        Returns:
            stats (dict): quantitative measures of one or all rows.

        Raises:
            IndexError: on row_label not found in data.
        """
        if row_label is None:
            try:
                stats = self.get_all_row_stats()
            except IndexError:
                raise
        else:
            stats = self.get_one_row_stats(row_label)
        return stats

    # ...............................................
    def get_one_row_stats(self, row_label):
        """Get a dictionary of statistics for the row with this row_label.

        Args:
            row_label: label on the row to gather stats for.

        Returns:
            stats (dict): quantitative measures of the row.

        Raises:
            IndexError: on row_label not found in data.

        Note:
            Inline comments are specific to a SUMMARY_TABLE_TYPES.SPECIES_DATASET_MATRIX
                with row/column/value = species/dataset/occ_count
        """
        # Get row (sparse array), and its index
        try:
            row, row_idx = self.get_vector_from_label(row_label, axis=0)
        except IndexError:
            raise
        # Largest/smallest Occurrence count for this Species, and column (dataset)
        # labels that contain it
        maxval, max_col_labels = self.get_extreme_val_labels_for_vector(
            row, axis=0, is_max=True)
        minval, min_col_labels = self.get_extreme_val_labels_for_vector(
            row, axis=0, is_max=False)

        stats = {
            self._keys[SNKeys.ROW_LABEL]: row_label,
            # Total Occurrences for this Species
            self._keys[SNKeys.ROW_TOTAL]: self.convert_np_vals_for_json(row.sum()),
            # Count of Datasets containing this Species
            self._keys[SNKeys.ROW_COUNT]: self.convert_np_vals_for_json(row.nnz),
            # Return min/max count in this species and datasets for that count
            self._keys[SNKeys.ROW_MIN_TOTAL]: minval,
            self._keys[SNKeys.ROW_MAX_TOTAL]: maxval,
            self._keys[SNKeys.ROW_MAX_TOTAL_LABELS]: max_col_labels
        }

        return stats

    # ...............................................
    def get_all_row_stats(self):
        """Return stats (min, max, mean, median) of totals and counts for all rows.

        Returns:
            all_row_stats (dict): counts and statistics about all rows.
            (numpy.ndarray): array of totals of all rows.
        """
        # Sum all rows to return a column (axis=1) of species totals
        all_totals = self._coo_array.sum(axis=1)
        # Min total and rows that contain it
        min_total = all_totals.min()
        min_total_number = self.count_val_in_vector(all_totals, min_total)
        # Max total and rows that contain that
        max_total = all_totals.max()
        # Get species names for largest number of occurrences
        max_total_labels = self.get_labels_for_val_in_vector(
            all_totals, max_total, axis=1)

        # For every row, get number of non-zero entries
        csr = self._coo_array.tocsr()
        all_counts = np.diff(csr.indptr)
        min_count = all_counts.min()
        min_count_number = self.count_val_in_vector(all_counts, min_count)
        max_count = all_counts.max()
        max_count_labels = self.get_labels_for_val_in_vector(
            all_counts, max_count, axis=1)

        # Count columns with at least one non-zero entry (all columns)
        row_count = self._coo_array.shape[0]
        all_row_stats = {
            # Count of other axis
            self._keys[SNKeys.ROWS_COUNT]: row_count,
            self._keys[SNKeys.ROWS_MIN_COUNT]:
                self.convert_np_vals_for_json(min_count),
            self._keys[SNKeys.ROWS_MIN_TOTAL_NUMBER]: min_count_number,

            self._keys[SNKeys.ROWS_MEAN_COUNT]:
                self.convert_np_vals_for_json(all_counts.mean()),
            self._keys[SNKeys.ROWS_MEDIAN_COUNT]:
                self.convert_np_vals_for_json(np.median(all_counts, axis=0)),

            self._keys[SNKeys.ROWS_MAX_COUNT]:
                self.convert_np_vals_for_json(max_count),
            self._keys[SNKeys.ROWS_MAX_COUNT_LABELS]: max_count_labels,

            # Total of values
            self._keys[SNKeys.ROWS_TOTAL]:
                self.convert_np_vals_for_json(all_totals.sum()),
            self._keys[SNKeys.ROWS_MIN_TOTAL]:
                self.convert_np_vals_for_json(min_total),
            self._keys[SNKeys.ROWS_MIN_TOTAL]: min_total_number,

            self._keys[SNKeys.ROWS_MEAN_TOTAL]:
                self.convert_np_vals_for_json(all_totals.mean()),
            self._keys[SNKeys.ROWS_MEDIAN_TOTAL]: self.convert_np_vals_for_json(
                np.median(all_totals, axis=0)[0, 0]),

            self._keys[SNKeys.ROWS_MAX_TOTAL]:
                self.convert_np_vals_for_json(max_total),
            self._keys[SNKeys.ROW_MAX_TOTAL_LABELS]: max_total_labels,
        }

        return all_row_stats

    # ...............................................
    def get_column_stats(self, col_label=None):
        """Return statistics for a one or all columns.

        Args:
            col_label (str): label of one column to get statistics for.

        Returns:
            stats (dict): quantitative measures of one or all columns.

        Raises:
            IndexError: on label not present in column header.
        """
        if col_label is None:
            stats = self.get_all_column_stats()
        else:
            try:
                stats = self.get_one_column_stats(col_label)
            except IndexError:
                raise
        return stats

    # ...............................................
    def get_one_column_stats(self, col_label):
        """Get a dictionary of statistics for this col_label or all columns.

        Args:
            col_label: label on the column to gather stats for.

        Returns:
            stats (dict): quantitative measures of the column.

        Raises:
            IndexError: on label not present in column header

        Note:
            Inline comments are specific to a SUMMARY_TABLE_TYPES.SPECIES_DATASET_MATRIX
                with row/column/value = species/dataset/occ_count
        """
        stats = {}
        # Get column (sparse array), and its index
        try:
            col, col_idx = self.get_vector_from_label(col_label, axis=1)
        except IndexError:
            raise
        # Largest/smallest occ count for dataset (column), and species (row) labels
        # containing that count.
        maxval, max_row_labels = self.get_extreme_val_labels_for_vector(
            col, axis=1, is_max=True)
        minval, min_row_labels = self.get_extreme_val_labels_for_vector(
            col, axis=1, is_max=False)

        stats[self._keys[SNKeys.COL_LABEL]] = col_label

        # Count of non-zero rows (Species) within this column (Dataset)
        stats[self._keys[SNKeys.COL_COUNT]] = self.convert_np_vals_for_json(col.nnz)
        # Total Occurrences for Dataset
        stats[self._keys[SNKeys.COL_TOTAL]] = self.convert_np_vals_for_json(col.sum())
        # Return min occurrence count in this dataset
        stats[self._keys[SNKeys.COL_MIN_TOTAL]] = self.convert_np_vals_for_json(minval)
        # Return number of species containing same minimum count (too many to list)
        stats[self._keys[SNKeys.COL_MIN_TOTAL_NUMBER]] = len(min_row_labels)
        # Return max occurrence count in this dataset
        stats[self._keys[SNKeys.COL_MAX_TOTAL]] = self.convert_np_vals_for_json(maxval)
        # Return species containing same maximum count
        stats[self._keys[SNKeys.COL_MAX_TOTAL_LABELS]] = max_row_labels

        return stats

    # ...............................................
    def get_all_column_stats(self):
        """Return stats (min, max, mean, median) of totals and counts for all columns.

        Returns:
            all_col_stats (dict): counts and statistics about all columns.
        """
        # Sum all rows for each column to return a row (numpy.matrix, axis=0)
        all_totals = self._coo_array.sum(axis=0)
        # Min total and columns that contain it
        min_total = all_totals.min()
        min_total_number = self.count_val_in_vector(all_totals, min_total)
        # Max total and columns that contain it
        max_total = all_totals.max()
        max_total_labels = self.get_labels_for_val_in_vector(
            all_totals, max_total, axis=0)

        # For every column, get number of non-zero rows
        csc = self._coo_array.tocsc()
        all_counts = np.diff(csc.indptr)
        # Min count and columns that contain that
        min_count = all_counts.min()
        min_count_number = self.count_val_in_vector(all_counts, min_count)
        # Max count and columns that contain that
        max_count = all_counts.max()
        max_count_labels = self.get_labels_for_val_in_vector(
            all_counts, max_count, axis=0)

        # Count rows with at least one non-zero entry (all rows)
        col_count = self._coo_array.shape[1]
        all_col_stats = {
            # Count of other axis
            self._keys[SNKeys.COLS_COUNT]: col_count,
            self._keys[SNKeys.COLS_MIN_COUNT]:
                self.convert_np_vals_for_json(min_count),
            self._keys[SNKeys.COLS_MIN_COUNT_NUMBER]: min_count_number,

            self._keys[SNKeys.COLS_MEAN_COUNT]:
                self.convert_np_vals_for_json(all_counts.mean()),
            self._keys[SNKeys.COLS_MEDIAN_COUNT]:
                self.convert_np_vals_for_json(np.median(all_counts, axis=0)),

            self._keys[SNKeys.COLS_MAX_COUNT]:
                self.convert_np_vals_for_json(max_count),
            self._keys[SNKeys.COLS_MAX_COUNT_LABELS]: max_count_labels,

            # Total occurrences
            self._keys[SNKeys.COLS_TOTAL]:
                self.convert_np_vals_for_json(all_totals.sum()),
            self._keys[SNKeys.COLS_MIN_TOTAL]:
                self.convert_np_vals_for_json(min_total),
            self._keys[SNKeys.COLS_MIN_TOTAL_NUMBER]: min_total_number,

            self._keys[SNKeys.COLS_MEAN_TOTAL]:
                self.convert_np_vals_for_json(all_totals.mean()),
            self._keys[SNKeys.COLS_MEDIAN_TOTAL]:
                self.convert_np_vals_for_json(np.median(all_totals, axis=1)[0, 0]),

            self._keys[SNKeys.COLS_MAX_TOTAL]: self.convert_np_vals_for_json(max_total),
            self._keys[SNKeys.COLS_MAX_TOTAL_LABELS]: max_total_labels,
        }
        return all_col_stats

    # ...............................................
    def get_totals(self, axis):
        """Get a list of value totals along the axis, down axis 0, across axis 1.

        Args:
            axis (int): Axis to sum.

        Returns:
            all_totals (list): list of values for the axis.
        """
        mtx = self._coo_array.sum(axis=axis)
        # Axis 0 produces a matrix shape (col_count,),
        #   1 row of values, total for each column
        # Axis 1 produces matrix shape (row_count,),
        #   1 column of values, total for each row
        if axis == 0:
            all_totals = mtx.tolist()
        elif axis == 1:
            all_totals = mtx.T.tolist()
        return all_totals

    # ...............................................
    def get_counts(self, axis):
        """Count non-zero values along the requested axis, down axis 0, across axis 1.

        Args:
            axis (int): Axis to count non-zero values for.

        Returns:
            all_counts (list): list of values for the axis.
        """
        if axis == 0:
            sp_arr = self._coo_array.tocsc()
        else:
            sp_arr = self._coo_array.tocsr()

        all_counts = np.diff(sp_arr.indptr)
        return all_counts

    # ...............................................
    def compare_column_to_others(self, col_label, agg_type=None):
        """Compare the number of rows and counts in rows to those of other columns.

        Args:
            col_label: label on the column to compare.
            agg_type: return stats on rows or values.  If None, return both.
                (options: "axis", "value", None)

        Returns:
            comparisons (dict): comparison measures
        """
        # Get this column stats
        stats = self.get_one_column_stats(col_label)
        # Show this column totals and counts compared to min, max, mean of all columns
        all_stats = self.get_all_column_stats()
        comparisons = {self._keys[SNKeys.COL_TYPE]: col_label}
        if agg_type in ("value", None):
            comparisons["Occurrences"] = {
                self._keys[SNKeys.COL_TOTAL]: stats[self._keys[SNKeys.COL_TOTAL]],
                self._keys[SNKeys.COLS_TOTAL]: all_stats[self._keys[SNKeys.COLS_TOTAL]],
                self._keys[SNKeys.COLS_MIN_TOTAL]:
                    all_stats[self._keys[SNKeys.COLS_MIN_TOTAL]],
                self._keys[SNKeys.COLS_MAX_TOTAL]:
                    all_stats[self._keys[SNKeys.COLS_MAX_TOTAL]],
                self._keys[SNKeys.COLS_MEAN_TOTAL]:
                    all_stats[self._keys[SNKeys.COLS_MEAN_TOTAL]],
                self._keys[SNKeys.COLS_MEDIAN_TOTAL]:
                    all_stats[self._keys[SNKeys.COLS_MEDIAN_TOTAL]]
            }
        if agg_type in ("axis", None):
            comparisons["Species"] = {
                self._keys[SNKeys.COL_COUNT]: stats[self._keys[SNKeys.COL_COUNT]],
                self._keys[SNKeys.COLS_COUNT]: all_stats[self._keys[SNKeys.COLS_COUNT]],
                self._keys[SNKeys.COLS_MIN_COUNT]:
                    all_stats[self._keys[SNKeys.COLS_MIN_COUNT]],
                self._keys[SNKeys.COLS_MAX_COUNT]:
                    all_stats[self._keys[SNKeys.COLS_MAX_COUNT]],
                self._keys[SNKeys.COLS_MEAN_COUNT]:
                    all_stats[self._keys[SNKeys.COLS_MEAN_COUNT]],
                self._keys[SNKeys.COLS_MEDIAN_COUNT]:
                    all_stats[self._keys[SNKeys.COLS_MEDIAN_COUNT]]
            }
        return comparisons

    # ...............................................
    def compare_row_to_others(self, row_label, agg_type=None):
        """Compare the number of columns and counts in columns to those of other rows.

        Args:
            row_label: label on the row to compare.
            agg_type: return stats on rows or values.  If None, return both.
                (options: "axis", "value", None)

        Returns:
            comparisons (dict): comparison measures
        """
        stats = self.get_one_row_stats(row_label)
        # Show this column totals and counts compared to min, max, mean of all columns
        all_stats = self.get_all_row_stats()
        comparisons = {self._keys[SNKeys.ROW_TYPE]: row_label}
        if agg_type in ("value", None):
            comparisons["Occurrences"] = {
                self._keys[SNKeys.ROW_TOTAL]: stats[self._keys[SNKeys.ROW_TOTAL]],
                self._keys[SNKeys.ROWS_TOTAL]: all_stats[self._keys[SNKeys.ROWS_TOTAL]],
                self._keys[SNKeys.ROWS_MIN_TOTAL]:
                    all_stats[self._keys[SNKeys.ROWS_MIN_TOTAL]],
                self._keys[SNKeys.ROWS_MAX_TOTAL]:
                    all_stats[self._keys[SNKeys.ROWS_MAX_TOTAL]],
                self._keys[SNKeys.ROWS_MEAN_TOTAL]:
                    all_stats[self._keys[SNKeys.ROWS_MEAN_TOTAL]],
                self._keys[SNKeys.ROWS_MEDIAN_TOTAL]:
                    all_stats[self._keys[SNKeys.ROWS_MEDIAN_TOTAL]],
            }
        if agg_type in ("axis", None):
            comparisons["Datasets"] = {
                self._keys[SNKeys.ROW_COUNT]: stats[self._keys[SNKeys.ROW_COUNT]],
                self._keys[SNKeys.ROWS_COUNT]: all_stats[self._keys[SNKeys.ROWS_COUNT]],
                self._keys[SNKeys.ROWS_MIN_COUNT]:
                    all_stats[self._keys[SNKeys.ROWS_MIN_COUNT]],
                self._keys[SNKeys.ROWS_MAX_COUNT]:
                    all_stats[self._keys[SNKeys.ROWS_MAX_COUNT]],
                self._keys[SNKeys.ROWS_MEAN_COUNT]:
                    all_stats[self._keys[SNKeys.ROWS_MEAN_COUNT]],
                self._keys[SNKeys.ROWS_MEDIAN_COUNT]:
                    all_stats[self._keys[SNKeys.ROWS_MEDIAN_COUNT]]
            }
        return comparisons

    # .............................................................................
    def _write_files(self, mtx_fname, meta_fname):
        # Save matrix to npz locally
        try:
            scipy.sparse.save_npz(mtx_fname, self._coo_array, compressed=True)
        except Exception as e:
            msg = f"Failed to write {mtx_fname}: {e}"
            self._logme(msg, log_level=ERROR)
            raise Exception(msg)

        # Save table data and categories to json locally
        metadata = deepcopy(self._table)
        metadata["row_categories"] = self._row_categ.categories.tolist()
        metadata["column_categories"] = self._col_categ.categories.tolist()
        metadata["value_fld"] = self.input_val_fld
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

    # .............................................................................
    def compress_to_file(self, local_path=TMP_PATH):
        """Compress this HeatmapMatrix to a zipped npz and json file.

        Args:
            local_path (str): Absolute path of local destination path

        Returns:
            zip_fname (str): Local output zip filename.

        Raises:
            Exception: on failure to write matrix and metadata files to zipfile.
        """
        # Always delete local files before compressing this data.
        [mtx_fname, meta_fname, zip_fname] = self._remove_expected_files(
            local_path=local_path)

        self._write_files(mtx_fname, meta_fname)

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
        """Uncompress a zipped HeatmapMatrix into a coo_array and row/column categories.

        Args:
            zip_filename (str): Filename of zipped sparse matrix data to uncompress.
            local_path (str): Absolute path of local destination path
            overwrite (bool): Flag indicating whether to use existing files unzipped
                from the zip_filename.

        Returns:
            sparse_coo (scipy.sparse.coo_array): Sparse Matrix containing data.
            row_categ (pandas.api.types.CategoricalDtype): row categories
            col_categ (pandas.api.types.CategoricalDtype): column categories
            table_type (specnet.tools.s2n.constants.SUMMARY_TABLE_TYPES): type of table
                data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on failure to uncompress files.
            Exception: on failure to load data from uncompressed files.

        Note:
            All filenames have the same basename with extensions indicating which data
                they contain. The filename contains a string like YYYY-MM-DD which
                indicates which GBIF data dump the statistics were built upon.
        """
        try:
            mtx_fname, meta_fname, table_type, datestr = cls._uncompress_files(
                zip_filename, local_path=local_path, overwrite=overwrite)
        except Exception:
            raise

        try:
            sparse_coo, meta_dict, row_categ, col_categ = cls.read_data(
                mtx_fname, meta_fname)
        except Exception:
            raise

        return sparse_coo, meta_dict, row_categ, col_categ, table_type, datestr

    # .............................................................................
    @classmethod
    def read_data(cls, mtx_filename, meta_filename):
        """Read HeatmapMatrix data files into a coo_array and row/column categories.

        Args:
            mtx_filename (str): Filename of scipy.sparse.coo_array data in npz format.
            meta_filename (str): Filename of JSON sparse matrix metadata.

        Returns:
            sparse_coo (scipy.sparse.coo_array): Sparse Matrix containing data.
            row_categ (pandas.api.types.CategoricalDtype): row categories
            col_categ (pandas.api.types.CategoricalDtype): column categories
            table_type (specnet.tools.s2n.constants.SUMMARY_TABLE_TYPES): type of table
                data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on unable to load NPZ file
            Exception: on unable to load JSON metadata file
            Exception: on missing row categories in JSON
            Exception: on missing column categories in JSON

        Note:
            All filenames have the same basename with extensions indicating which data
                they contain. The filename contains a string like YYYY-MM-DD which
                indicates which GBIF data dump the statistics were built upon.
        """
        # Read sparse matrix from npz file
        try:
            sparse_coo = scipy.sparse.load_npz(mtx_filename)
        except Exception as e:
            raise Exception(f"Failed to load {mtx_filename}: {e}")

        # Read JSON dictionary as string
        try:
            meta_dict = cls.load_metadata(meta_filename)
        except Exception:
            raise

        # Parse metadata into objects for matrix construction
        try:
            row_catlst = meta_dict.pop("row_categories")
        except KeyError:
            raise Exception(f"Missing row categories in {meta_filename}")
        else:
            row_categ = CategoricalDtype(row_catlst, ordered=True)
        try:
            col_catlst = meta_dict.pop("column_categories")
        except KeyError:
            raise Exception(f"Missing column categories in {meta_filename}")
        else:
            col_categ = CategoricalDtype(col_catlst, ordered=True)

        return sparse_coo, meta_dict, row_categ, col_categ
