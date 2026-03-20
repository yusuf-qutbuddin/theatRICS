import multiprocessing as mp
import tkinter as tk
from theatrics.gui_app import ModularRICSGUI
from theatrics.utils.mp_utils import set_single_threaded_blas, setup_multiprocessing


def main():
    set_single_threaded_blas()
    setup_multiprocessing("spawn", force=True)

    root = tk.Tk()
    app = ModularRICSGUI(root)

    def on_closing():
        # optional: terminate running worker processes cleanly
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()