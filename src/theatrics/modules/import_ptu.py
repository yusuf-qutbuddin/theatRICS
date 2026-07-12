import SFCS_module as sfcs 

sfcs.read_file(r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/RawLine.ptu', 1)





# import export_rics as exp

# dictionary = exp.read_ptu_metadata(r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/RawImage.ptu')

# print(dictionary)

"""
debug_ptu_linescan.py

Diagnostic script for PicoQuant line-scan PTU files.
Does NOT use CLSMImage — reads the raw TTTR stream directly.

Usage:
    python debug_ptu_linescan.py /path/to/linescan.ptu
"""

# import sys
# import numpy as np
# import tttrlib

# if len(sys.argv) > 1:
#     filepath = sys.argv[1]
# else:
#     filepath = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/RawLine.ptu'

# print("=" * 60)
# print(f"File: {filepath}")
# print("=" * 60)

# # ── load raw TTTR data ────────────────────────────────────────────
# print("\n[1] Loading TTTR data...")
# tttr_data = tttrlib.TTTR(filepath)
# print(f"    OK")

# # ── header data ───────────────────────────────────────────────────
# print("\n[2] Relevant header tags...")
# hd = tttr_data.header.data
# keys_of_interest = [
#     "ImgHdr_PixX",
#     "ImgHdr_PixY",
#     "ImgHdr_MaxFrames",
#     "ImgHdr_PixResol",
#     "ImgHdr_TimePerPixel",
#     "ImgHdr_LineStart",
#     "ImgHdr_LineStop",
#     "ImgHdr_Frame",
#     "ImgHdr_Ident",
#     "ImgHdr_BiDirect",
#     "ImgHdr_ScanDirection",
#     "Measurement_SubMode",
#     "Measurement_Mode",
#     "MeasDesc_GlobalResolution",
#     "MeasDesc_AcquisitionTime",
#     "TTResult_NumberOfRecords",
#     "HW_Markers",
# ]
# for k in keys_of_interest:
#     val = hd.get(k, "NOT FOUND")
#     print(f"    {k:35s} : {val}")

# # ── raw TTTR arrays ───────────────────────────────────────────────
# print("\n[3] Raw TTTR stream info...")
# macro_times      = tttr_data.macro_times
# micro_times      = tttr_data.micro_times
# routing_channels = tttr_data.routing_channels
# event_types      = tttr_data.event_types

# print(f"    Total records      : {len(macro_times)}")
# print(f"    macro_times dtype  : {macro_times.dtype}")
# print(f"    micro_times dtype  : {micro_times.dtype}")
# print(f"    routing_channels   : unique = {np.unique(routing_channels)}")
# print(f"    event_types        : unique = {np.unique(event_types)}")

# # ── event type breakdown ──────────────────────────────────────────
# print("\n[4] Event type breakdown...")
# for et in np.unique(event_types):
#     n = int(np.sum(event_types == et))
#     print(f"    event_type={et} : {n} events")

# # ── marker events ─────────────────────────────────────────────────
# # In PicoQuant PTU files, markers are typically stored as
# # special routing channel values (e.g. negative or > 15)
# # OR as event_type != 0
# print("\n[5] Marker events (event_type != 0)...")
# marker_mask = event_types != 0
# n_markers   = int(np.sum(marker_mask))
# print(f"    Total marker events: {n_markers}")

# if n_markers > 0:
#     marker_channels = routing_channels[marker_mask]
#     marker_times    = macro_times[marker_mask]
#     print(f"    Marker channel values (unique): {np.unique(marker_channels)}")
#     print(f"    First 20 marker events:")
#     for i in range(min(20, n_markers)):
#         idx = np.where(marker_mask)[0][i]
#         print(f"        idx={idx:8d}  macro_time={macro_times[idx]:12d}"
#               f"  channel={routing_channels[idx]:4d}"
#               f"  event_type={event_types[idx]}")

# # ── photon events ─────────────────────────────────────────────────
# print("\n[6] Photon events (event_type == 0)...")
# photon_mask = event_types == 0
# n_photons   = int(np.sum(photon_mask))
# print(f"    Total photon events: {n_photons}")
# if n_photons > 0:
#     ph_channels = routing_channels[photon_mask]
#     print(f"    Photon routing channels (unique): {np.unique(ph_channels)}")

# # ── timing resolution ─────────────────────────────────────────────
# print("\n[7] Timing info from header...")
# macro_res = hd.get("MeasDesc_GlobalResolution", [None])[0]
# print(f"    MeasDesc_GlobalResolution (s/tick): {macro_res}")

# if macro_res is not None and n_markers > 0:
#     # estimate line period from consecutive marker events of same type
#     marker_channels_arr = routing_channels[marker_mask]
#     marker_times_arr    = macro_times[marker_mask]

#     for ch_val in np.unique(marker_channels_arr):
#         ch_mask  = marker_channels_arr == ch_val
#         ch_times = marker_times_arr[ch_mask]
#         if len(ch_times) > 2:
#             diffs    = np.diff(ch_times.astype(np.float64)) * float(macro_res)
#             median_d = float(np.median(diffs))
#             print(f"    Marker channel={ch_val}: "
#                   f"n={len(ch_times)}  "
#                   f"median interval={median_d*1e3:.4f} ms  "
#                   f"({1.0/median_d:.2f} Hz)")

# # ── try get_line_duration and get_pixel_duration from header ──────
# print("\n[8] Header timing methods...")
# try:
#     ld = tttr_data.header.get_line_duration()
#     print(f"    header.get_line_duration()  = {ld}")
# except Exception as e:
#     print(f"    header.get_line_duration()  failed: {e}")

# try:
#     pd = tttr_data.header.get_pixel_duration()
#     print(f"    header.get_pixel_duration() = {pd}")
# except Exception as e:
#     print(f"    header.get_pixel_duration() failed: {e}")

# # ── macro_time range ──────────────────────────────────────────────
# print("\n[9] Macro time range...")
# print(f"    macro_times min : {macro_times.min()}")
# print(f"    macro_times max : {macro_times.max()}")
# if macro_res is not None:
#     total_s = float(macro_times.max()) * float(macro_res)
#     print(f"    Total duration  : {total_s:.4f} s  ({total_s*1e3:.2f} ms)")

# # ── first 50 events ───────────────────────────────────────────────
# print("\n[10] First 50 raw events...")
# print(f"    {'idx':>6}  {'macro_time':>14}  {'micro_time':>12}"
#       f"  {'channel':>8}  {'event_type':>10}")
# for i in range(min(50, len(macro_times))):
#     print(f"    {i:6d}  {macro_times[i]:14d}  {micro_times[i]:12d}"
#           f"  {routing_channels[i]:8d}  {event_types[i]:10d}")

# print("\n" + "=" * 60)
# print("Debug complete. Please share this full output.")
# print("=" * 60)