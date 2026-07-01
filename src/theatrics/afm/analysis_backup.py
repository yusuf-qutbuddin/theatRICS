import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy import ndimage
from scipy.signal import savgol_filter
from AFMReader.jpk import load_jpk

# ============================================================
# LOAD
# ============================================================

def load_jpk_qi(filepath, channel='height_trace'):
    """
    Load a .jpk-qi-image file using AFMReader.
    AFMReader applies the internal JPK scaling (slope + offset)
    and for 'height_trace' returns values directly in nm.

    Returns
    -------
    height_nm     : 2D numpy array, heights in nm
    pixel_size_nm : float, physical size of one pixel in nm
    """
    height_nm, pixel_size_nm = load_jpk(filepath, channel=channel,
                                         flip_image=True)
    print(f"Loaded    : {filepath}")
    print(f"Channel   : {channel}")
    print(f"Shape     : {height_nm.shape}")
    print(f"Px size   : {pixel_size_nm:.3f} nm/px")
    print(f"Scan size : {height_nm.shape[1] * pixel_size_nm / 1000:.3f} µm")
    print(f"Heights   : {height_nm.min():.3f} – {height_nm.max():.3f} nm")
    return height_nm, pixel_size_nm


# ============================================================
# PROFILE EXTRACTION
# ============================================================

def extract_line_profile(height_data, pixel_size_nm, start_px, end_px,
                         n_points=300):
    """
    Sample the height map along a straight line between two pixel
    coordinates using bilinear interpolation.

    Returns distances (nm) and heights (nm) along the line.
    """
    x0, y0 = start_px
    x1, y1 = end_px
    x_coords = np.linspace(x0, x1, n_points)
    y_coords = np.linspace(y0, y1, n_points)
    heights  = ndimage.map_coordinates(
        height_data, [y_coords, x_coords], order=1
    )
    total_dist_nm = np.sqrt((x1-x0)**2 + (y1-y0)**2) * pixel_size_nm
    distances     = np.linspace(0, total_dist_nm, n_points)
    return distances, heights


# ============================================================
# BASELINE CORRECTION
# ============================================================

def compute_baseline_and_correction(distances_nm, heights_nm,
                                    h_start, h_end,
                                    smooth_window=15, smooth_poly=3):
    """
    Subtract a linear baseline defined by the two heights the user
    clicked on (both guaranteed to be on bare membrane).

    The baseline is the straight line from h_start at distance=0
    to h_end at distance=total_length. Subtracting it puts the
    membrane plane at zero everywhere along the profile, even if
    the sample is slightly tilted.

    Also applies Savitzky-Golay smoothing before subtraction to
    reduce pixel-level noise without distorting peak shapes.

    Returns
    -------
    h_smooth : smoothed raw profile (nm)
    baseline : the linear baseline (nm)
    h_adj    : baseline-corrected profile, membrane = 0 (nm)
    """
    n = len(heights_nm)

    if n > smooth_window:
        h_smooth = savgol_filter(heights_nm, smooth_window, smooth_poly)
    else:
        h_smooth = heights_nm.copy()

    baseline = h_start + (h_end - h_start) * distances_nm / distances_nm[-1]
    h_adj    = h_smooth - baseline

    return h_smooth, baseline, h_adj


# ============================================================
# CONTACT POINT DETECTION & WETTING ANGLE
# ============================================================

def find_contact_and_angles(distances_nm, h_adj, n_fit_points=15):
    """
    Find where the condensate meets the membrane and compute the
    wetting angle at each contact point.

    Contact point definition
    ------------------------
    The first index from the left where h_adj exceeds 5% of the
    peak height, and the first index from the right where h_adj
    exceeds 5% of the peak height.

    This threshold-based approach is more robust than a zero-crossing
    because the corrected profile may not go exactly to zero at the
    edges due to noise and the finite smoothing window.

    Wetting angle
    -------------
    A straight line is fitted through n_fit_points starting AT the
    contact point and going INTO the condensate (up the rising edge
    on the left, up the falling edge on the right).

        slope = dh/dx  [nm/nm]  (dimensionless)
        theta = arctan(|slope|) in degrees

    The angle is measured from the membrane plane (horizontal) to
    the condensate surface at the contact line.

    Parameters
    ----------
    distances_nm  : 1D array, distance along profile in nm
    h_adj         : 1D array, baseline-corrected heights in nm
    n_fit_points  : number of points used to fit the tangent line

    Returns
    -------
    dict with contact indices, slopes, and angles, or None if
    contact points cannot be found.
    """
    n        = len(h_adj)
    peak_idx = int(np.argmax(h_adj))
    h_peak   = h_adj[peak_idx]

    if h_peak <= 0:
        return None

    threshold = 0.05 * h_peak   # 5% of peak height

    # Left contact: first point from left above threshold
    left_contact = None
    for i in range(0, peak_idx):
        if h_adj[i] > threshold:
            left_contact = i
            break

    # Right contact: first point from right above threshold
    right_contact = None
    for i in range(n - 1, peak_idx, -1):
        if h_adj[i] > threshold:
            right_contact = i
            break

    if left_contact is None or right_contact is None:
        return None

    def slope_at(ci, direction, npts):
        """
        Fit a line through npts points starting at contact index ci,
        going in direction (+1 rightward, -1 leftward) into condensate.
        """
        idx = np.arange(ci, ci + direction * npts, direction)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) < 2:
            return np.nan
        return np.polyfit(distances_nm[idx], h_adj[idx], 1)[0]

    left_slope  = slope_at(left_contact,  +1, n_fit_points)
    right_slope = slope_at(right_contact, -1, n_fit_points)

    angles = {
        'left_contact':  left_contact,
        'right_contact': right_contact,
        'n_fit_points':  n_fit_points,
        'h_peak':        h_peak,
        'threshold':     threshold,
    }

    if not np.isnan(left_slope):
        angles['theta_left_deg']  = np.degrees(np.arctan(abs(left_slope)))
        angles['left_slope']      = left_slope
        angles['left_contact_nm'] = distances_nm[left_contact]

    if not np.isnan(right_slope):
        angles['theta_right_deg']  = np.degrees(np.arctan(abs(right_slope)))
        angles['right_slope']      = right_slope
        angles['right_contact_nm'] = distances_nm[right_contact]

    if 'theta_left_deg' in angles and 'theta_right_deg' in angles:
        angles['theta_mean_deg'] = np.mean([angles['theta_left_deg'],
                                            angles['theta_right_deg']])
    return angles


# ============================================================
# INTERACTIVE VIEWER
# ============================================================

class InteractiveProfileSelector:

    def __init__(self, height_data, pixel_size_nm, scan_size_um):
        self.height_data   = height_data 
        self.pixel_size_nm = pixel_size_nm
        self.scan_size_um  = scan_size_um
        self.ny, self.nx   = height_data.shape

        self.points        = []   # list of (x_px, y_px, h_nm)
        self.profile_lines = []
        self.all_results   = []   # list of result dicts
        self.n_fit_points  = 15

        self._build_figure()

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.suptitle(
            "Click two points on the AFM image to extract a line profile  |  "
            "R = reset last  |  C = clear all  |  Q = quit",
            fontsize=10
        )
        self.ax_afm     = self.fig.add_axes([0.03, 0.18, 0.44, 0.75])
        self.ax_profile = self.fig.add_axes([0.55, 0.18, 0.42, 0.75])

        ax_reset = self.fig.add_axes([0.03, 0.04, 0.09, 0.06])
        ax_clear = self.fig.add_axes([0.14, 0.04, 0.09, 0.06])
        self.btn_reset = Button(ax_reset, 'Reset last')
        self.btn_clear = Button(ax_clear, 'Clear all')
        self.btn_reset.on_clicked(self._reset_last)
        self.btn_clear.on_clicked(self._clear_all)

        ax_slider = self.fig.add_axes([0.55, 0.06, 0.35, 0.04])
        self.slider = Slider(ax_slider, 'Fit points',
                             valmin=3, valmax=60,
                             valinit=self.n_fit_points, valstep=1)
        self.slider.on_changed(self._on_slider)

        self._draw_afm()
        self._init_profile_panel()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)
        plt.show()

    def _draw_afm(self):
        self.ax_afm.cla()
        self.ax_afm.imshow(
            self.height_data, cmap='afmhot', aspect='equal',
            extent=[0, self.scan_size_um, self.scan_size_um, 0],
            interpolation='bilinear'
        )
        self.ax_afm.set_xlabel('x (µm)', fontsize=11)
        self.ax_afm.set_ylabel('y (µm)', fontsize=11)
        self.ax_afm.set_title('AFM height image', fontsize=11)
        for seg in self.profile_lines:
            self.ax_afm.plot(seg[0], seg[1], 'c-', linewidth=1.8)
        self.fig.canvas.draw_idle()

    def _init_profile_panel(self):
        self.ax_profile.cla()
        self.ax_profile.set_xlabel('Distance (nm)', fontsize=11)
        self.ax_profile.set_ylabel('Height (nm)',   fontsize=11)
        self.ax_profile.set_title(
            'Click: membrane background → across condensate → membrane',
            fontsize=11)
        self.ax_profile.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self.ax_afm or event.button != 1:
            return

        x_px = np.clip(event.xdata / self.scan_size_um * self.nx,
                       0, self.nx - 1)
        y_px = np.clip(event.ydata / self.scan_size_um * self.ny,
                       0, self.ny - 1)

        # Sample the exact height at the clicked pixel
        h_nm = float(ndimage.map_coordinates(
            self.height_data, [[y_px], [x_px]], order=1
        )[0])

        self.points.append((x_px, y_px, h_nm))

        self.ax_afm.plot(event.xdata, event.ydata, '+',
                         color='cyan', markersize=12, markeredgewidth=2)
        self.fig.canvas.draw_idle()

        if len(self.points) == 2:
            self._process_profile()
            self.points = []

    def _on_key(self, event):
        if   event.key in ('r', 'R'): self._reset_last(None)
        elif event.key in ('c', 'C'): self._clear_all(None)
        elif event.key in ('q', 'Q'): plt.close(self.fig)

    def _on_slider(self, val):
        self.n_fit_points = int(val)
        if not self.all_results:
            return
        last = self.all_results[-1]
        new_angles = find_contact_and_angles(
            last['distances_nm'], last['h_adj'],
            n_fit_points=self.n_fit_points
        )
        last['angles']       = new_angles
        last['n_fit_points'] = self.n_fit_points
        self._plot_profile(last)

    # ----------------------------------------------------------
    # Core pipeline
    # ----------------------------------------------------------

    def _process_profile(self):
        (x0, y0, h0), (x1, y1, h1) = self.points

        seg_x = [x0 * self.pixel_size_nm / 1000,
                 x1 * self.pixel_size_nm / 1000]
        seg_y = [y0 * self.pixel_size_nm / 1000,
                 y1 * self.pixel_size_nm / 1000]
        self.profile_lines.append((seg_x, seg_y))

        dist, heights = extract_line_profile(
            self.height_data, self.pixel_size_nm,
            (x0, y0), (x1, y1)
        )
        h_smooth, baseline, h_adj = compute_baseline_and_correction(
            dist, heights, h_start=h0, h_end=h1
        )
        angles = find_contact_and_angles(
            dist, h_adj, n_fit_points=self.n_fit_points
        )

        entry = {
            'distances_nm':  dist,
            'heights_raw':   heights,
            'h_smooth':      h_smooth,
            'baseline':      baseline,
            'h_adj':         h_adj,
            'h_start':       h0,
            'h_end':         h1,
            'angles':        angles,
            'n_fit_points':  self.n_fit_points,
        }
        self.all_results.append(entry)
        self._plot_profile(entry)
        self._draw_afm()

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------

    def _plot_profile(self, entry):
        ax = self.ax_profile
        ax.cla()

        dist     = entry['distances_nm']
        raw      = entry['heights_raw']
        baseline = entry['baseline']
        h_adj    = entry['h_adj']
        n        = len(dist)
        nfp      = entry['n_fit_points']
        angles   = entry['angles']

        # Raw profile
        ax.plot(dist, raw,
                color='steelblue', linewidth=1.5, alpha=0.4,
                label='Raw')

        # Baseline
        ax.plot(dist, baseline,
                color='saddlebrown', linewidth=2, linestyle='--',
                label=f"Baseline "
                      f"({entry['h_start']:.1f}→{entry['h_end']:.1f} nm)")

        # Baseline-corrected profile
        ax.plot(dist, h_adj,
                color='royalblue', linewidth=2,
                label='Corrected (membrane = 0)')
        ax.fill_between(dist, 0, h_adj, where=(h_adj > 0),
                        alpha=0.15, color='royalblue')

        # Membrane reference
        ax.axhline(0, color='saddlebrown', linewidth=1,
                   linestyle='-', alpha=0.4)

        # Contact points and tangent lines
        if angles is not None:
            for side, ck, sk, tk, color, direction in [
                ('left',  'left_contact',  'left_slope',  'theta_left_deg',
                 'tomato',         +1),
                ('right', 'right_contact', 'right_slope', 'theta_right_deg',
                 'mediumseagreen', -1),
            ]:
                if ck not in angles or sk not in angles:
                    continue

                ci    = angles[ck]
                x_c   = dist[ci]
                slope = angles[sk]
                theta = angles.get(tk, float('nan'))

                # Contact point at membrane level
                ax.plot(x_c, 0, 'o', color=color, markersize=10, zorder=6,
                        label=f'{side.capitalize()} θ = {theta:.1f}°')

                # Points used for slope fit
                fit_idx = np.arange(ci, ci + direction * nfp, direction)
                fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n)]
                ax.plot(dist[fit_idx], h_adj[fit_idx], 'o',
                        color=color, markersize=5, alpha=0.9, zorder=5)

                # Tangent line through contact point
                span  = dist[-1] * 0.25
                x_ext = np.array([x_c - span, x_c + span])
                y_ext = slope * (x_ext - x_c)
                ax.plot(x_ext, y_ext, '--', color=color, linewidth=2)

            tl = angles.get('theta_left_deg',  float('nan'))
            tr = angles.get('theta_right_deg', float('nan'))
            tm = angles.get('theta_mean_deg',  float('nan'))
            title = (f'θ_left = {tl:.1f}°  |  '
                     f'θ_right = {tr:.1f}°  |  '
                     f'θ_mean = {tm:.1f}°')
        else:
            title = (f'Contact points not found — '
                     f'ensure line crosses condensate fully  '
                     f'(peak = {h_adj.max():.2f} nm)')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Distance (nm)', fontsize=11)
        ax.set_ylabel('Height (nm)',   fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # Button callbacks
    # ----------------------------------------------------------

    def _reset_last(self, _event):
        if self.profile_lines: self.profile_lines.pop()
        if self.all_results:   self.all_results.pop()
        self.points = []
        self._draw_afm()
        self._init_profile_panel()
        if self.all_results:
            self._plot_profile(self.all_results[-1])

    def _clear_all(self, _event):
        self.profile_lines.clear()
        self.all_results.clear()
        self.points = []
        self._draw_afm()
        self._init_profile_panel()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    filepath = r'/fs/pool/pool-schwille-user/Qutbuddin_Yusuf/_Protocols/RICS_fit/test_data/afm/first.jpk-qi-image'

    height_data, pixel_size_nm = load_jpk_qi(filepath, channel='height_trace')
    scan_size_um = height_data.shape[1] * pixel_size_nm / 1000

    InteractiveProfileSelector(height_data, pixel_size_nm, scan_size_um)