from AFMReader.jpk import load_jpk
import numpy as np

def inspect_jpk(filepath):
    """
    Inspect a .jpk-qi-image file using AFMReader and print
    everything we can find out about it.
    """
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")

    # --- Try loading with different common channel names ---
    channel_names = [
        'height',
        'height_trace',
        'height_retrace', 
        'measuredHeight_trace',
        'measuredHeight',
        'amplitude',
        'heightTrace',
    ]

    successful_channels = []

    for channel in channel_names:

        try:
            result = load_jpk(filepath, channel=channel)
            print(f"\n[OK] channel='{channel}' loaded successfully")
            print(f"     type of result : {type(result)}")

            # result might be a tuple, array, or custom object
            # print everything we can
            if isinstance(result, tuple):
                print(f"     tuple length   : {len(result)}")
                for i, item in enumerate(result):
                    print(f"     item[{i}]        : type={type(item)}")
                    if isinstance(item, np.ndarray):
                        print(f"                      shape={item.shape}")
                        print(f"                      dtype={item.dtype}")
                        print(f"                      min={item.min():.4e}")
                        print(f"                      max={item.max():.4e}")
                    else:
                        print(f"                      value={item}")

            elif isinstance(result, np.ndarray):
                print(f"     shape          : {result.shape}")
                print(f"     dtype          : {result.dtype}")
                print(f"     min            : {result.min():.4e}")
                print(f"     max            : {result.max():.4e}")

            else:
                # custom object — print all attributes
                print(f"     attributes     : {dir(result)}")
                for attr in dir(result):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(result, attr)
                            if not callable(val):
                                print(f"     .{attr} = {val}")
                        except Exception:
                            pass

            successful_channels.append(channel)

        except Exception as e:
            print(f"[--] channel='{channel}' failed: {e}")

    print(f"\n{'='*60}")
    print(f"Successfully loaded channels: {successful_channels}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    filepath = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/afm/first.jpk-qi-image'
    filepath = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/afm/second.jpk-qi-image'

    inspect_jpk(filepath)