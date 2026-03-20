import os

def get_files_from_folder(folder_path, extension, suffix=""):
    out = []
    for f in os.listdir(folder_path):
        fname, ext = os.path.splitext(f)
        if ext.lower() == extension.lower() and fname.lower().endswith(suffix.lower()):
            out.append(os.path.join(folder_path, f))
    return out