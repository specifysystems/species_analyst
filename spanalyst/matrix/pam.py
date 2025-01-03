"""Matrix of sites as rows, species as columns, values are presence or absence (1/0)."""
from copy import deepcopy
from memory_profiler import profile
import numpy as np
import os
import pandas as pd
from zipfile import ZipFile

from spanalyst.common.constants import (
    CSV_DELIMITER, CSV_EXTENSION, JSON_EXTENSION, STATISTICS_TYPE, SUMMARY, TMP_PATH,
    ZIP_EXTENSION
)
from spanalyst.matrix.heatmap import HeatmapMatrix


# .............................................................................
class PAM(HeatmapMatrix):
    """Class for analyzing presence/absence of aggregator0 x species (aggregator1)."""

    # ...........................
    def __init__(
            self, binary_coo_array, min_presence_count, table_type, datestr,
            row_category, column_category, dim0, dim1):
        """Constructor for species by region/analysis_dim comparisons.

        Args:
            binary_coo_array (scipy.sparse.coo_array): A 2d sparse array with
                presence (1) or absence (0) values for one dimension (i.e. region) rows
                (axis 0) by the species dimension columns (axis 1) to use for analyses.
            min_presence_count (int): minimum value to be considered presence.
            table_type (specnet.tools.s2n.constants.SUMMARY_TABLE_TYPES): type of
                aggregated data
            datestr (str): date of the source data in YYYY_MM_DD format.
            row_category (pandas.api.types.CategoricalDtype): ordered row labels used
                to identify axis 0/rows.
            column_category (pandas.api.types.CategoricalDtype): ordered column labels
                used to identify axis 1/columns.
            dim0 (specnet.common.constants.ANALYSIS_DIM): dimension for axis 0, rows
            dim1 (spanalyst.common.constants.SPECIES): dimension for axis 1, columns,
                always species dimension in specnet PAM matrices

        Raises:
            Exception: on values other than 0 or 1.

        Note:
            By definition, a Presence-Absence Matrix is site x species.  This
                implementation defines `site` as any type of geographic (state, county,
                Indian lands, Protected Areas) or other classification (dataset,
                organization, US-RIIS status) where every occurrence contains at most
                one `site` value. Some statistics may assume that all occurrences will
                contain a site value, but this implementation does not enforce that
                assumption.
        """
        # Check PAM is binary (0/1)
        tmp = binary_coo_array > 1
        if tmp.count_nonzero() > 0:
            raise Exception("Only 0 and 1 are allowed in a Presence-Absence Matrix")
        # Check PAM is numpy.int8
        if binary_coo_array.dtype != np.int8:
            binary_coo_array = binary_coo_array.astype(np.int8)

        cmp_pam_coo_array, cmp_row_categ, cmp_col_categ = self._remove_zeros(
            binary_coo_array, row_category, column_category)
        self._min_presence = min_presence_count
        val_fld = "presence"

        # Populate this on computation, with keys used in filename construction
        self.stats_matrices = {}
        for key in STATISTICS_TYPE.all():
            self.stats_matrices[key] = None

        HeatmapMatrix.__init__(
            self, cmp_pam_coo_array, table_type, datestr, cmp_row_categ, cmp_col_categ,
            dim0, dim1, val_fld)

    # # ...........................
    # @classmethod
    # def init_from_heatmap(cls, heatmap, min_presence_count):
    #     """Create a sparse matrix of rows by columns containing values from a table.
    #
    #     Args:
    #         heatmap (spanalyst.matrix.heatmap_matrix.HeatmapMatrix): Matrix of occurrence
    #             counts for sites (or other dimension), rows, by species, columns.
    #         min_presence_count (int): Minimum occurrence count for a species to be
    #             considered present at that site.
    #
    #     Returns:
    #         pam (spanalyst.matrix.pam_matrix.PAM): matrix of
    #             sites (rows, axis=0) by species (columnns, axis=1), with binary values
    #             indicating presence/absence.
    #     """
    #     # Apply minimum value filter; converts to CSR format
    #     bool_csr_array = heatmap._coo_array >= min_presence_count
    #     pam_csr_array = bool_csr_array.astype(np.int8)
    #     # Go back to COO format
    #     pam_coo_array = pam_csr_array.tocoo()
    #
    #     pam = PAM(
    #         pam_coo_array, min_presence_count, heatmap.table_type, heatmap.datestr,
    #         heatmap.row_category, heatmap.column_category,
    #         heatmap.y_dimension, heatmap.x_dimension)
    #     return pam

    # ...........................
    @classmethod
    def init_from_heatmap(cls, heatmap, min_presence_count):
        """Create a sparse matrix of rows by columns containing values from a table.

        Args:
            heatmap (spanalyst.matrix.heatmap.HeatmapMatrix): Matrix of occurrence
                counts for sites (or other dimension), rows, by species, columns.
            min_presence_count (int): Minimum occurrence count for a species to be
                considered present at that site.

        Returns:
            pam (spanalyst.matrix.pam_matrix.PAM): matrix of
                sites (rows, axis=0) by species (columnns, axis=1), with binary values
                indicating presence/absence.
        """
        filtered_heatmap = heatmap.filter(
            min_count=min_presence_count)

        # Convert to boolean (all True because pre-filtered)
        bool_csr_array = filtered_heatmap._coo_array > 0
        # Convert to binary
        pam_csr_array = bool_csr_array.astype(np.int8)
        # Go back to COO format
        pam_coo_array = pam_csr_array.tocoo()

        dim0, dim1 = heatmap.dimensions
        pam_table_type = SUMMARY.get_table_type("pam", dim0, dim1)

        pam = PAM(
            pam_coo_array, min_presence_count, pam_table_type, heatmap.datestr,
            heatmap.row_category, heatmap.column_category,
            heatmap.y_dimension, heatmap.x_dimension)
        return pam

    # .............................................................................
    @classmethod
    def get_stats_filenames(cls, base_filename, local_path=TMP_PATH):
        """Get the statistics compressed filename and its matrix and metadata contents.

        Args:
            base_filename (str): Base filename (without path or extension) for the
                compressed file of statistics matrices and metadata.
            local_path (str): Absolute path of local destination path

        Returns:
            expected_files (dict): Dictionary of keys: zip_fname and statistics names,
                and values, a list of the zip filename or matrix and metadata filenames.
        """
        expected_file_dict = {}
        zip_fname = f"{base_filename}{ZIP_EXTENSION}"
        if local_path is not None:
            zip_fname = os.path.join(local_path, zip_fname)
        expected_file_dict["zip_fname"] = [zip_fname]

        for stats_type in STATISTICS_TYPE.all():
            fname = f"{base_filename}_{stats_type}"
            mtx_fname = f"{fname}{CSV_EXTENSION}"
            meta_fname = f"{fname}{JSON_EXTENSION}"
            if local_path is not None:
                mtx_fname = os.path.join(local_path, mtx_fname)
                meta_fname = os.path.join(local_path, meta_fname)
            expected_file_dict[stats_type] = [mtx_fname, meta_fname]

        return expected_file_dict

    # .............................................................................
    def _write_files(self, stats_type, stats_df, mtx_fname, meta_fname):
        # Always delete local files before compressing this data.
        try:
            stats_df.to_csv(mtx_fname, sep=CSV_DELIMITER)
        except Exception as e:
            msg = f"Failed to write {mtx_fname}: {e}"
            raise Exception(msg)

        # Save table data and categories to json locally
        metadata = deepcopy(self._table)
        metadata["statistics_type"] = stats_type
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
    def compress_stats_to_file(self, local_path=TMP_PATH):
        """Compress all statistics matrices to zipped npz and json files, then zip all.

        Args:
            local_path (str): Absolute path of local destination path

        Returns:
            zip_fname (str): Local output zip filename.

        Raises:
            Exception: on failure to write matrix and metadata files to zipfile.
        """
        # Always delete local zipfile before compressing this data.
        base_filename = self._table["fname"]
        expected_file_dict = self.get_stats_filenames(
            base_filename, local_path=local_path)
        for fname_lst in expected_file_dict.values():
            for fn in fname_lst:
                if os.path.exists(fn):
                    os.remove(fn)

        input_fnames = []
        # Save each matrix with metadata locally
        for stats_type, stats_df in self.stats_matrices.items():
            if stats_df is not None:
                # Dictionaries stats_matrices and expected files use stat_name as key
                mtx_fname, meta_fname = expected_file_dict[stats_type]
                # Write matrix and metadata files
                self._write_files(stats_type, stats_df, mtx_fname, meta_fname)

                input_fnames.extend((mtx_fname, meta_fname))

        # Compress all matrices and metadata
        zip_fname = expected_file_dict["zip_fname"][0]
        try:
            self._compress_files(input_fnames, zip_fname)
        except Exception:
            raise

        return zip_fname

    # .............................................................................
    @classmethod
    def uncompress_zipped_data(cls, zip_filename, local_path=TMP_PATH):
        """Uncompress a zipped SparseMatrix into a coo_array and row/column categories.

        Args:
            zip_filename (str): Filename of output data to write to S3.
            local_path (str): Absolute path of local destination path

        Returns:
            statistics_dict (dict of pandas.DataFrame): dict of statistic types and
                corresponding dataframes containing statistics data.
            meta_dict (dict): metadata for the set of matrices
            table_type (aws.aws_constants.SUMMARY_TABLE_TYPES): type of table data
            datestr (str): date string in format YYYY_MM_DD

        Raises:
            Exception: on failure to uncompress files.
            Exception: on failure to load data from uncompressed files.
        """
        base_filename, _ext = os.path.splitext(os.path.basename(zip_filename))
        try:
            out_filenames, table_type, datestr = cls._uncompress_files(
                zip_filename, local_path)
        except Exception:
            raise

        # TODO: parse filenames instead of look for expected
        stats_fname_dict = {}
        expected_file_dict = cls.get_stats_filenames(
            base_filename, local_path=local_path)
        for key, fnames in expected_file_dict.items():
            if key != "zip_fname":
                for fn in fnames:
                    if fn in out_filenames:
                        try:
                            stats_fname_dict[key].append(fn)
                        except KeyError:
                            stats_fname_dict[key] = [fn]

        # Read matrix data from local files
        try:
            stats_data_dict, stats_meta_dict = cls.read_data(stats_fname_dict)
        except Exception:
            raise

        return stats_data_dict, stats_meta_dict, table_type, datestr

    # .............................................................................
    @classmethod
    def _uncompress_files(cls, zip_filename, local_path):
        """Uncompress a zipped set of PAM statistics matrices and their metadata.

        Args:
            zip_filename (str): Filename of output data to write to S3.
            local_path (str): Absolute path of local destination path

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

        Note:
            Always delete existing files before uncompress
        """
        if not os.path.exists(zip_filename):
            raise Exception(f"Missing file {zip_filename}")
        try:
            table_type, datestr = SUMMARY.get_tabletype_datestring_from_filename(
                zip_filename)
        except Exception:
            raise

        # Delete all pre-existing matrix, metadata files
        table = SUMMARY.get_table(table_type, datestr=datestr)
        base_filename = table["fname"]
        expected_file_dict = cls.get_stats_filenames(
            base_filename, local_path=local_path)
        for key, fname_lst in expected_file_dict.items():
            if key != "zip_fname":
                for fn in fname_lst:
                    if os.path.exists(fn):
                        os.remove(fn)

        # Unzip to local dir
        with ZipFile(zip_filename, mode="r") as archive:
            fnames = archive.namelist()
            archive.extractall(path=local_path)

        out_filenames = [os.path.join(local_path, fn) for fn in fnames]

        return out_filenames, table_type, datestr

    # .............................................................................
    @classmethod
    def read_data(cls, statistics_fname_dict):
        """Read SummaryMatrix data files into a dataframe and metadata dictionary.

        Args:
            statistics_fname_dict (dict): Statistics type with list of filenames of
                pandas.DataFrame data in csv format and JSON metadata.

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
        stats_data_dict = {}
        stats_meta_dict = {}
        for stat_type, filename_lst in statistics_fname_dict.items():
            for fn in filename_lst:
                try:
                    if fn.endswith(CSV_EXTENSION):
                        dataframe = pd.read_csv(fn, sep=CSV_DELIMITER, index_col=0)
                        stats_data_dict[stat_type] = dataframe
                    elif fn.endswith(JSON_EXTENSION):
                        meta_dict = cls.load_metadata(fn)
                        stats_meta_dict[stat_type] = meta_dict
                except Exception as e:
                    raise Exception(f"Failed to load {fn}: {e}")

        return stats_data_dict, stats_meta_dict

    # ...........................
    def num_species(self):
        """Return number of species in the array (on the x/1 axis).

        Returns:
            (int): Number of species (values on the x/1 axis)
        """
        return self._coo_array.shape[1]

    # ...........................
    def num_sites(self):
        """Return number of `sites` in the array (on the y/0 axis).

        Returns:
            (int): Number of `sites` (values on the y/0 axis)
        """
        return self._coo_array.shape[0]

    # ...........................
    def calc_covariance_stats(self):
        """Calculate the site sigma metric and species sigma metric.

        Postcondition:
            The stats_matrices dictionary is populated with:
                cov_site_mtx (pandas.DataFrame): a matrix
                cov_species_mtx (pandas.DataFrame): a matrix

        Note:
            Both of these computations are too memory intensive for a very large
            number of sites or species.

        TODO: make covariance computation more efficient.
        """
        cov_site_mtx = self.sigma_sites()
        cov_species_mtx = self.sigma_species()
        self.stats_matrices[STATISTICS_TYPE.SIGMA_SITE] = cov_site_mtx
        self.stats_matrices[STATISTICS_TYPE.SIGMA_SPECIES] = cov_species_mtx

    # ...........................
    def calc_diversity_stats(self):
        """Calculate diversity statistics.

        Postcondition:
            The stats_matrices dictionary is populated with:
                diversity_matrix (pandas.DataFrame): a matrix with 1 column for each
                    statistic, and one row containing the values for each statistic.
        """
        diversity_stats = [
            ('num sites', self.num_sites),
            ('num species', self.num_species),
            # ('c-score', self.c_score),
            ('lande', self.lande),
            ('legendre', self.legendre),
            ('whittaker', self.whittaker),
            ]
        data = {}
        for name, func in diversity_stats:
            data[name] = func()
        diversity_matrix = pd.DataFrame(data=data, index=["value"])
        self.stats_matrices[STATISTICS_TYPE.DIVERSITY] = diversity_matrix

    # ...........................
    def calc_site_stats(self):
        """Calculate site-based statistics.

        Postcondition:
            The stats_matrices dictionary is populated with:
                site_stats_matrix (pandas.DataFrame): a matrix with 1 column for each
                    statistic, and one row for each site.
        """
        site_matrix_stats = [
            ("alpha", self.alpha, None),
            ("alpha proportional", self.alpha_proportional, "alpha"),
            ("phi", self.phi, None),
            ("phi average proportional", self.phi_average_proportional, "phi"),
        ]
        site_index = self.row_category.categories
        data = {}
        for name, func, input_name in site_matrix_stats:
            try:
                input_data = data[input_name]
            except KeyError:
                data[name] = func()
            else:
                data[name] = func(input_data)

        site_stats_matrix = pd.DataFrame(data=data, index=site_index)
        self.stats_matrices[STATISTICS_TYPE.SITE] = site_stats_matrix

    # ...........................
    def calc_species_stats(self):
        """Calculate species-based statistics.

        Postcondition:
            The stats_matrices dictionary is populated with:
                site_stats_matrix (pandas.DataFrame): a matrix with 1 column for each
                    statistic, and one row for each species.
        """
        species_matrix_stats = [
            ("omega", self.omega, None),
            ("omega_proportional", self.omega_proportional, "omega"),
            ("psi", self.psi, None),
            ("psi_average_proportional", self.psi_average_proportional, "psi"),
        ]
        species_index = self._col_categ.categories
        data = {}
        for name, func, input_name in species_matrix_stats:
            try:
                input_data = data[input_name]
            except KeyError:
                data[name] = func()
            else:
                data[name] = func(input_data)

        species_stats_matrix = pd.DataFrame(data=data, index=species_index)
        self.stats_matrices[STATISTICS_TYPE.SPECIES] = species_stats_matrix

    # .............................................................................
    # Diversity metrics
    # .............................................................................
    # TODO: test the matrices created by sigma functions within these diversity stats
    @profile
    def schluter_species_variance_ratio(self):
        """Calculate Schluter's species variance ratio.

        Returns:
            schl_sp (float): The Schluter species variance ratio for the PAM.
        """
        sigma_species_, _hdrs = self.sigma_species()
        trace = sigma_species_.trace()
        schl_sp = float(sigma_species_.sum()) / trace
        return schl_sp

    # .............................................................................
    @profile
    def schluter_site_variance_ratio(self):
        """Calculate Schluter's site variance ratio.

        Returns:
            schl_site (float): The Schluter site variance ratio for the PAM.
        """
        sigma_sites_, _hdrs = self.sigma_sites()
        trace = sigma_sites_.trace()
        schl_site = float(sigma_sites_.sum()) / trace
        return schl_site

    # .............................................................................
    @profile
    def whittaker(self):
        """Calculate Whittaker's beta diversity metric for a PAM.

        Returns:
            whit (float): Whittaker's beta diversity for the PAM.
        """
        omg = None
        omega_prp = self.omega_proportional(omg)
        whit = self.num_species() / omega_prp.sum()
        return whit

    # .............................................................................
    @profile
    def lande(self):
        """Calculate Lande's beta diversity metric for a PAM.

        Returns:
            land (float): Lande's beta diversity for the PAM.
        """
        # range size (count) per species
        omg = None
        omega_prp = self.omega_proportional(omg)
        land = self.num_species() - omega_prp.sum()
        return land

    # .............................................................................
    @profile
    def legendre(self):
        """Calculate Legendre's beta diversity metric for a PAM.

        Returns:
            leg (float): Legendre's beta diversity for the PAM.
        """
        # range size (count) per species
        omega_fl = self.omega().astype(float)
        leg = omega_fl.sum() - (omega_fl ** 2).sum() / self.num_sites()
        return leg

    # ...........................
    @profile
    def c_score(self):
        """Calculate the checkerboard score for the PAM.

        Returns:
            chk (float): The checkerboard score for the PAM.

        TODO: Test and integrate into diversity stats
        """
        temp = 0.0
        # Cache these so we don't recompute
        omega_ = self.omega()
        num_species_ = self.num_species()
        pam = self._coo_array.todense()

        for i in range(num_species_):
            for j in range(i, num_species_):
                num_shared = len(np.where(np.sum(pam[:, [i, j]], axis=1) == 2)[0])
                p_1 = omega_[i] - num_shared
                p_2 = omega_[j] - num_shared
                temp += p_1 * p_2
        chk = 2 * temp / (num_species_ * (num_species_ - 1))
        return chk

    # .............................................................................
    # Species metrics
    # .............................................................................
    @profile
    def omega(self):
        """Calculate the range `size` (number of sites) per species.

        Returns:
            sp_range_size_vct (numpy.ndarray): 1D ndarray of range `sizes`
                (site-count), one element for each species (axis 1) of PAM,
                datatype int32.

        Note:
            function assumes all `sites` (analysis dimension) are equal size.
        """
        sp_range_size_vct = self._coo_array.sum(axis=0)
        sp_range_size_vct = sp_range_size_vct.astype(np.int32)
        return sp_range_size_vct

    # ...........................
    @profile
    def omega_proportional(self, omg=None):
        """Calculate the mean proportional range size of each species.

        Args:
            omg (numpy.ndarray): array of number of sites per species.

        Returns:
            omega_prp (numpy.ndarray): A row of the range sizes for each species
                proportional to the site count.
        """
        if omg is None:
            omg = self.omega()
        omega_prp = omg.astype(float) / self.num_sites()
        return omega_prp

    # .............................................................................
    @profile
    def psi(self):
        """Calculate the range richness of each species.

        Returns:
            psi_vct (numpy.ndarray): 1D array of range richness for the sites of each
                species.
        """
        pam = self._coo_array.todense(order="C")
        # vector of species count for each site
        alpha_vct = self.alpha()
        sp_range_richness_vct = alpha_vct.dot(pam)
        return sp_range_richness_vct

    # .............................................................................
    @profile
    def psi_average_proportional(self, psi_vct=None):
        """Calculate the mean proportional range richness of each species.

        Args:
            psi_vct (numpy.ndarray): array of range richness per species

        Returns:
            psi_vct (numpy.ndarray): 1D array of range richness for the sites of each
                species proportional to all species' range size.
        """
        if psi_vct is None:
            psi_vct = self.psi()
        sp_range_size_vector = self.num_species() * self.omega()
        psi_avg_prop = psi_vct.astype(float) / sp_range_size_vector
        return psi_avg_prop

    # .............................................................................
    # Site-based statistics
    # .............................................................................
    @profile
    def alpha(self):
        """Calculate alpha diversity, the number of species in each site.

        Returns:
            sp_count_vct (numpy.ndarray): 1D ndarray of species count for each
                site in the PAM, datatype int32.
        """
        sp_count_vct = self._coo_array.sum(axis=1)
        sp_count_vct = sp_count_vct.astype(np.int32)
        return sp_count_vct

    # .............................................................................
    @profile
    def alpha_proportional(self, alpha_vct=None):
        """Calculate proportional alpha diversity.

        Args:
            alpha_vct (numpy.ndarray): 1D array of species count per site.

        Returns:
            alpha_prop_vct (numpy.ndarray): 1D array, row, of proportional alpha
                diversity values for each site in the PAM.
        """
        if alpha_vct is None:
            alpha_vct = self.alpha()
        alpha_prop_vct = alpha_vct.astype(float) / self.num_species()
        return alpha_prop_vct

    # .............................................................................
    @profile
    def phi(self):
        """Calculate phi, the range size per site.

        Returns:
            phi_vct (numpy.ndarray): A 1D array of the average range size of all
                species present at each site in the PAM.
        """
        pam = self._coo_array.todense(order="C")
        omega_vct = self.omega()
        phi_vct = pam.dot(omega_vct)
        return phi_vct

    # .............................................................................
    @profile
    def phi_average_proportional(self, phi_vct=None):
        """Calculate proportional range size per site.

        Args:
            phi_vct (numpy.ndarray): array of average range size

        Returns:
            phi_avg_prop_mtx (numpy.ndarray): A 1D matrix of the value of
                the sum of the range sizes for species present at each site in the PAM
                proportional to the number of species in that site.
        """
        if phi_vct is None:
            phi_vct = self.phi()
        alpha_ = self.alpha()
        phi_avg_prop_mtx = phi_vct / (self.num_sites() * alpha_)
        return phi_avg_prop_mtx

    # .............................................................................
    # Covariance metrics
    # .............................................................................
    @profile
    def sigma_sites(self):
        """Compute the site sigma metric for a PAM.

        Returns:
            mtx (numpy.ndarray): Matrix of covariance of composition of sites.
            headers (dict): categories for axis 0 and 1 headers.

        TODO: find a more efficient implementation, difficult on very large datasets
        """
        pam = self._coo_array.todense(order="C")
        site_by_site = pam.dot(pam.T).astype(float)
        # vector of species count per site
        alpha_prop = self.alpha_proportional()
        mtx = (site_by_site / self.num_species()) - np.outer(alpha_prop, alpha_prop)
        # Output is sites x sites, so use site headers for column headers too
        headers = {
            "0": deepcopy(self.row_category),
            "1": deepcopy(self.column_category)
        }
        return mtx, headers

    # .............................................................................
    @profile
    def sigma_species(self):
        """Compute the species sigma metric for a PAM.

        Returns:
            mtx (numpy.ndarray): Matrix of covariance of composition of species.
            headers (dict): categories for axis 0 and 1 headers.

        TODO: find a more efficient implementation, difficult on very large datasets
        """
        pam = self._coo_array.todense(order="C")
        species_by_site = pam.T.dot(pam).astype(float)
        omega_prop = self.omega_proportional(omg=None)
        mtx = (species_by_site / self.num_sites()) - np.outer(omega_prop, omega_prop)
        # Output is species x species, so use species headers for row headers too
        headers = {
            "0": deepcopy(self.row_category),
            "1": deepcopy(self.column_category)
        }
        return mtx, headers
