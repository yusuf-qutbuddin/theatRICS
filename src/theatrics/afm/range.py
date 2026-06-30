from AFMReader.jpk import load_jpk, _load_jpk_tags
import numpy as np

filepath = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/afm/first.jpk-qi-image'

image_raw, pixel_size_nm = load_jpk(filepath, channel='height_trace', flip_image=True)
config = _load_jpk_tags()
print(config)
print(f"raw min  : {image_raw.min():.6e}")
print(f"raw max  : {image_raw.max():.6e}")
print(f"raw mean : {image_raw.mean():.6e}")
print(f"pixel_size_nm : {pixel_size_nm}")

# Test all plausible conversions and print what height range they give
conversions = {
    'as-is'    : image_raw,
    '* 1e9'    : image_raw * 1e9,
    '* 1e6'    : image_raw * 1e6,
    '* 1e3'    : image_raw * 1e3,
    '/ 1e3'    : image_raw / 1e3,
    '/ 1e6'    : image_raw / 1e6,
    '/ 1e9'    : image_raw / 1e9,
}

print("\nConversion test (peak-to-peak range after each conversion):")
for label, arr in conversions.items():
    ptp = arr.max() - arr.min()
    print(f"  {label:10s} : min={arr.min():.4e}  max={arr.max():.4e}  "
          f"peak-to-peak={ptp:.4e} nm?")

print(f"\nExpected peak-to-peak for your data: ~100–500 nm")
print(f"Match the conversion that gives that range.")