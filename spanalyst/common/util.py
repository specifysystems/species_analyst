"""Common tools to use either locally or on EC2."""
# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
import csv
import datetime as DT
import glob
import os
import sys
import traceback

from spanalyst.common.constants import ENCODING, SHP_EXT, SHP_EXTENSIONS


# ----------------------------------------------------
def get_today_str():
    """Get a string representation of the current date.

    Returns:
        date_str(str): string representing date in YYYY-MM-DD format.
    """
    n = DT.datetime.now()
    date_str = f"{n.year}_{n.month:02d}_{n.day:02d}"
    return date_str


# ----------------------------------------------------
def get_current_datadate_str():
    """Get a string representation of the first day of the current month.

    Returns:
        date_str(str): string representing date in YYYY-MM-DD format.
    """
    n = DT.datetime.now()
    date_str = f"{n.year}_{n.month:02d}_01"
    # date_str = "2024_08_01"
    return date_str


# ----------------------------------------------------
def get_previous_datadate_str():
    """Get a string representation of the first day of the previous month.

    Returns:
        date_str(str): string representing date in YYYY-MM-DD format.
    """
    n = DT.datetime.now()
    yr = n.year
    mo = n.month - 1
    if n.month == 0:
        mo = 12
        yr -= 1
    date_str = f"{yr}_{mo:02d}_01"
    return date_str


# --------------------------------------------------------------------------------------
# Tools for experimentation
# --------------------------------------------------------------------------------------

# ----------------------------------------------------upload_to_s3
def _print_inst_info(reservation):
    resid = reservation["ReservationId"]
    inst = reservation["Instances"][0]
    print(f"ReservationId: {resid}")
    name = temp_id = None
    try:
        tags = inst["Tags"]
    except Exception:
        pass
    else:
        for t in tags:
            if t["Key"] == "Name":
                name = t["Value"]
            if t["Key"] == "aws:ec2launchtemplate:id":
                temp_id = t["Value"]
    ip = inst["PublicIpAddress"]
    state = inst["State"]["Name"]
    print(f"Instance name: {name}, template: {temp_id}, IP: {ip}, state: {state}")


# # ...............................................
# def delete_file(file_name, delete_dir=False):
#     """Delete file if it exists, optionally delete newly empty directory.
#
#     Args:
#         file_name (str): full path to the file to delete
#         delete_dir (bool): flag - True to delete parent directory if it becomes empty
#
#     Returns:
#         True if file was not found, or file was successfully deleted.  If
#             file deletion results in an empty parent directory, directory is also
#             successfully deleted.
#         False if failed to delete file (and parent directories).
#     """
#     success = True
#     msg = ''
#     if file_name is None:
#         msg = "Cannot delete file 'None'"
#     else:
#         pth, _ = os.path.split(file_name)
#         if file_name is not None and os.path.exists(file_name):
#             try:
#                 os.remove(file_name)
#             except Exception as e:
#                 success = False
#                 msg = 'Failed to remove {}, {}'.format(file_name, str(e))
#             if delete_dir and len(os.listdir(pth)) == 0:
#                 try:
#                     os.removedirs(pth)
#                 except Exception as e:
#                     success = False
#                     msg = 'Failed to remove {}, {}'.format(pth, str(e))
#     return success, msg

# ...............................................
def delete_file(full_filename, delete_dir=False):
    """Delete file if it exists, delete directory if it becomes empty.

    Args:
        full_filename: Full path to filename to check.
        delete_dir: True to delete the parent directory if empty after file deletion.

    Returns:
        success: True on successful deletion, False on failure.
        msg: An error message if success is False

    Note:
        If file is shapefile, delete all related files
    """
    success = True
    msg = ""
    if full_filename is None:
        msg = "Cannot delete file `None`"
    else:
        pth, _ = os.path.split(full_filename)
        if full_filename is not None and os.path.exists(full_filename):
            base, ext = os.path.splitext(full_filename)
            if ext == SHP_EXT:
                similar_file_names = glob.glob(f"{base}.*")
                try:
                    for sim_file_name in similar_file_names:
                        _, sim_ext = os.path.splitext(sim_file_name)
                        if sim_ext in SHP_EXTENSIONS:
                            os.remove(sim_file_name)
                except Exception as e:
                    success = False
                    msg = f"Failed to remove {sim_file_name}, {e}"
            else:
                try:
                    os.remove(full_filename)
                except Exception as e:
                    success = False
                    msg = f"Failed to remove {full_filename}, {e}"
            if delete_dir and len(os.listdir(pth)) == 0:
                try:
                    os.removedirs(pth)
                except Exception as e:
                    success = False
                    msg = f"Failed to remove {pth}, {e}"
    return success, msg


# ...............................................
def ready_filename(fullfilename, overwrite=True):
    """Delete file if it exists, optionally delete newly empty directory.

    Args:
        fullfilename (str): full path of the file to check
        overwrite (bool): flag indicating to delete the file if it already exists

    Returns:
        boolean: True if file does not yet exist, or file was successfully deleted.  If
            file deletion results in an empty parent directory, directory is also
            successfully deleted.
        False if failed to delete file (and parent directories).

    Raises:
        PermissionError: if unable to delete existing file when overwrite is true
        Exception: on other delete errors or failure to create directories
        PermissionError: if unable to create missing directories
        Exception: on other mkdir errors
        Exception: on failure to create directories
    """
    is_ready = True
    if os.path.exists(fullfilename):
        if overwrite:
            try:
                delete_file(fullfilename)
            except PermissionError:
                raise
            except Exception as e:
                raise Exception('Unable to delete {} ({})'.format(fullfilename, e))
        else:
            is_ready = False
    else:
        pth, _ = os.path.split(fullfilename)
        try:
            os.makedirs(pth)
        except FileExistsError:
            pass
        except PermissionError:
            raise
        except Exception:
            raise

        if not os.path.isdir(pth):
            raise Exception('Failed to create directories {}'.format(pth))

    return is_ready


# .............................................................................
def get_csv_reader(datafile, delimiter, encoding):
    """Get a CSV DictReader that can handle encoding.

    Args:
        datafile: filename for CSV input.
        delimiter: field separator for input
        encoding: file encoding for input

    Returns:
        reader: a CSV Reader
        f: an open file object.

    Raises:
        Exception: on failure to read or open the datafile.
    """
    try:
        f = open(datafile, "r", encoding=encoding)
        reader = csv.reader(
            f, delimiter=delimiter, escapechar="\\", quoting=csv.QUOTE_NONE)
    except Exception as e:
        raise Exception(f"Failed to read or open {datafile}, ({e})")
    else:
        print(f"Opened file {datafile} for read")
    return reader, f


# .............................................................................
def get_csv_writer(datafile, delimiter, encoding, fmode="w"):
    """Get a CSV writer that can handle encoding.

    Args:
        datafile: filename for CSV output.
        delimiter: field separator for output
        encoding: file encoding for output
        fmode: mode for writing, either write ("w") or append ("a")

    Returns:
        writer: a CSV Writer
        f: an open file object.

    Raises:
        Exception: on invalid file mode.
        Exception: on failure to read or open the datafile.
    """
    if fmode not in ("w", "a"):
        raise Exception("File mode must be 'w' (write) or 'a' (append)")

    csv.field_size_limit(sys.maxsize)
    try:
        f = open(datafile, fmode, encoding=encoding)
        writer = csv.writer(
            f, escapechar="\\", delimiter=delimiter, quoting=csv.QUOTE_NONE)
    except Exception as e:
        raise Exception(f"Failed to read or open {datafile}, ({e})")
    else:
        print(f"Opened file {datafile} for write")
    return writer, f


# .............................................................................
def get_csv_dict_writer(
        csvfile, header, delimiter, fmode="w", encoding=ENCODING, extrasaction="ignore",
        overwrite=True):
    """Create a CSV dictionary writer and write the header.

    Args:
        csvfile (str): output CSV filename for writing
        header (list): header for output file
        delimiter (str): field separator
        fmode (str): Write ('w') or append ('a')
        encoding (str): Encoding for output file
        extrasaction (str): Action to take if there are fields in a record dictionary
            not present in fieldnames
        overwrite (bool): True to delete an existing file before write

    Returns:
        writer (csv.DictWriter) ready to write
        f (file handle)

    Raises:
        Exception: on invalid file mode
        Exception: on failure to create a DictWriter
        FileExistsError: on existing file if overwrite is False
    """
    if fmode not in ("w", "a"):
        raise Exception("File mode must be 'w' (write) or 'a' (append)")
    if ready_filename(csvfile, overwrite=overwrite):
        csv.field_size_limit(sys.maxsize)
        try:
            f = open(csvfile, fmode, newline="", encoding=encoding)
        except Exception as e:
            raise e
        else:
            writer = csv.DictWriter(
                f, fieldnames=header, delimiter=delimiter, extrasaction=extrasaction)
            writer.writeheader()
        return writer, f
    else:
        raise FileExistsError


# .............................................................................
def get_csv_dict_reader(
        csvfile, delimiter, fieldnames=None, encoding=ENCODING, quote_none=False,
        restkey="rest"):
    """Create a CSV dictionary reader from a file with a fieldname header.

    Args:
        csvfile (str): output CSV file for reading
        delimiter (char): delimiter between fields
        fieldnames (list): strings with corrected fieldnames, cleaned of illegal
            characters, for use with records.
        encoding (str): type of encoding
        quote_none (bool): True opens csvfile with QUOTE_NONE, False opens with
            QUOTE_MINIMAL
        restkey (str): fieldname for extra fields in a record not present in header

    Returns:
        rdr (csv.DictReader): DictReader ready to read
        f (object): open file handle

    Raises:
        FileNotFoundError: on missing csvfile
        PermissionError: on improper permissions on csvfile
    """
    csv.field_size_limit(sys.maxsize)

    if quote_none is True:
        quoting = csv.QUOTE_NONE
    else:
        quoting = csv.QUOTE_MINIMAL

    try:
        #  If csvfile is a file object, it should be opened with newline=""
        f = open(csvfile, "r", newline="", encoding=encoding)
    except FileNotFoundError:
        raise
    except PermissionError:
        raise

    if fieldnames is not None:
        rdr = csv.DictReader(
            f, fieldnames=fieldnames, quoting=quoting, delimiter=delimiter,
            restkey=restkey)
    else:
        rdr = csv.DictReader(f, quoting=quoting, delimiter=delimiter, restkey=restkey)

    return rdr, f


# ..........................
def get_traceback():
    """Get the traceback for this exception.

    Returns:
        trcbk: traceback of steps executed before an exception
    """
    exc_type, exc_val, this_traceback = sys.exc_info()
    tb = traceback.format_exception(exc_type, exc_val, this_traceback)
    tblines = []
    cr = "\n"
    for line in tb:
        line = line.rstrip(cr)
        parts = line.split(cr)
        tblines.extend(parts)
    trcbk = cr.join(tblines)
    return trcbk


# ...............................................
def combine_errinfo(errinfo1, errinfo2):
    """Combine 2 dictionaries with keys `error`, `warning` and `info`.

    Args:
        errinfo1: dictionary of errors
        errinfo2: dictionary of errors

    Returns:
        dictionary of errors
    """
    errinfo = {}
    for key in ("error", "warning", "info"):
        try:
            lst = errinfo1[key]
        except KeyError:
            lst = []
        try:
            lst2 = errinfo2[key]
        except KeyError:
            lst2 = []

        if lst or lst2:
            lst.extend(lst2)
            errinfo[key] = lst
    return errinfo


# ...............................................
def add_errinfo(errinfo, key, val_lst):
    """Add to a dictionary with keys `error`, `warning` and `info`.

    Args:
        errinfo: dictionary of errors
        key: error type, `error`, `warning` or `info`
        val_lst: error message or list of errors

    Returns:
        updated dictionary of errors
    """
    if errinfo is None:
        errinfo = {}
    if key in ("error", "warning", "info"):
        if isinstance(val_lst, str):
            val_lst = [val_lst]
        try:
            errinfo[key].extend(val_lst)
        except KeyError:
            errinfo[key] = val_lst
    return errinfo


# ......................................................
def prettify_object(print_obj):
    """Format an object for output.

    Args:
        print_obj (obj): Object to pretty print in output

    Returns:
        formatted string representation of object

    Note: this splits a string containing spaces in a list to multiple strings in the
        list.
    """
    # Used only in local debugging
    from io import StringIO
    from pprint import pp

    strm = StringIO()
    pp(print_obj, stream=strm)
    obj_str = strm.getvalue()
    return obj_str
