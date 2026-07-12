import os
import glob

def get_files_from_folder(folder_path, extension, suffix="", recursive=True):
    """
    Return a list of full file paths matching `extension` (and optional
    filename `suffix`) inside `folder_path`.

    Parameters
    ----------
    folder_path : str
        Root folder to search.
    extension : str
        File extension to match, including the dot (e.g. '.tif', '.czi').
    suffix : str
        Optional filename suffix (before the extension) that the
        filename must end with, e.g. 'RICScorr' to match
        '..._RICScorr.tif'. Case-insensitive. Empty string matches
        everything.
    recursive : bool
        If True (default), searches `folder_path` AND all of its
        subfolders, at any depth -- e.g. measurement folders each
        containing several repetitions:

            folder/
                measurement_1/
                    rep1.tif
                    rep2.tif
                measurement_2/
                    rep1.tif

        If False, only searches the top-level `folder_path` itself
        (legacy behaviour).

    Returns
    -------
    list[str]  full paths of matching files, sorted alphabetically.
    """
    if recursive:
        pattern = os.path.join(folder_path, "**", f"*{extension}")
        candidates = glob.glob(pattern, recursive=True)
    else:
        candidates = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
        ]

    out = []
    for full_path in candidates:
        if not os.path.isfile(full_path):
            continue
        fname, ext = os.path.splitext(os.path.basename(full_path))
        if ext.lower() == extension.lower() and fname.lower().endswith(suffix.lower()):
            out.append(full_path)

    return sorted(out)