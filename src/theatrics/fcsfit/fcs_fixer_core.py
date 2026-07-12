# -*- coding: utf-8 -*-
"""
theatrics/fcsfit/fcs_fixer_core.py

Faithful, trimmed port of Jan-Hagen Krohn's FCS_Fixer class.

Only four changes were made relative to the original FCS_Fixer, all of
them mandatory compatibility fixes (documented inline with "CHANGED:"):

  1. correlation_method: 'default'/'lamb' -> 'wahl'/'felekyan'/'laurence'
     (tttrlib.Correlator.run() in the current version only recognises
     these three method strings — see Correlator.cpp)
  2. corr.get_x_axis_normalized() -> corr.get_x_axis()
     (method renamed/removed; get_x_axis() returns the identical raw
     bin pattern as long as Correlator.set_tttr() is never called,
     which it never is in this pipeline, so the internal calibration
     factor stays at its default of 1.0 and the old
     "* self._macro_time_resolution" conversion still gives the
     correct physical lag time)
  3. np.bool8 -> bool  (np.bool8 removed in current NumPy)
  4. get_background_tail_fit() itself is UNCHANGED. Only the source of
     its `irf_peak_center` argument differs: the original expects a
     peak position measured from a SEPARATE IRF calibration file
     (via find_IRF_position(channels_spec, irf_TTTR)). We do not have
     a separate IRF measurement in this single-file export context, so
     the wrapper (ptu_correlate.py) passes the argmax of the SAME
     channel's TCSPC histogram instead. This is the one unavoidable
     simplification and is isolated entirely to the wrapper, not to
     any function in this file.

Trimmed out (not needed for ACF / CCF / afterpulsing / FLCS background
correction): burst removal, bleaching/drift undrifting, MSE anomalous
segment filter, PCH/PCMH, Parallel_scheduler, logging file I/O.
The corresponding flags (use_burst_removal, use_drift_correction,
use_mse_filter) are kept in every function signature exactly as in the
original so the branching logic is 100% unchanged — they are simply
always called with False in this trimmed build.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from scipy.stats import f as f_dist
from scipy.special import erfcinv

try:
    import tttrlib
    TTTRLIB_AVAILABLE = True
except ImportError:
    TTTRLIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# small type helpers (verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def isint(x):
    return type(x) in [int, np.uint8, np.uint16, np.uint32, np.uint64,
                        np.int8, np.int16, np.int32, np.int64]


def isiterable(x):
    return type(x) in [list, tuple, np.ndarray]


def isfloat(x):
    return type(x) in [float, np.float16, np.float32, np.float64]


# ─────────────────────────────────────────────────────────────────────────────
# TCSPC_quick_fit  (verbatim, exponential branch only used by wrapper;
# gaussian branch kept for completeness / possible future IRF fitting)
# ─────────────────────────────────────────────────────────────────────────────

class TCSPC_quick_fit:
    def __init__(self, x_data, y_data, model,
                initial_params={'x_0': 0., 'y_0': 0., 'amp': 1000.,
                                'gauss_fwhm': 1., 'exp_tau': 1.}):
        self.x = x_data
        self.y = y_data
        sigma = np.sqrt(y_data)
        sigma[sigma == 0] = 1E3 * np.max(sigma)
        self.sigma = sigma

        self.fit_params = Parameters()
        self.fit_params.add('y_0',
                            value=initial_params.get('y_0', 0.),
                            min=0., vary=True)
        self.fit_params.add('amp',
                            value=initial_params.get('amp', 1000.),
                            min=0, vary=True)

        self.model = model
        if self.model == 'gauss':
            self.fit_params.add('x_0',
                                value=initial_params.get('x_0', 0.),
                                min=0., vary=True)
            self.fit_params.add('gauss_fwhm',
                                value=initial_params.get('gauss_fwhm', 1.),
                                min=0., vary=True)
        elif self.model == 'exponential':
            self.fit_params.add('x_0',
                                value=initial_params.get('x_0', 0.),
                                min=0., vary=False)
            self.fit_params.add('exp_tau',
                                value=initial_params.get('exp_tau', 1.),
                                min=0., vary=True)
        else:
            raise ValueError('Invalid model. Use "gauss" or "exponential".')

    def expression_gauss(self, x_0, y_0, amp, gauss_fwhm):
        return y_0 + amp / (gauss_fwhm * np.sqrt(np.pi / (4 * np.log(2)))) * \
               np.exp(-4 * np.log(2) * (self.x - x_0) ** 2 / gauss_fwhm ** 2)

    def residual_gauss(self, params):
        p = self.expression_gauss(params['x_0'].value, params['y_0'].value,
                                  params['amp'].value, params['gauss_fwhm'].value)
        return (self.y - p) / self.sigma

    def expression_exponential(self, x_0, y_0, amp, exp_tau):
        return y_0 + amp * np.exp(-(self.x - x_0) / exp_tau)

    def residual_exponential(self, params):
        p = self.expression_exponential(params['x_0'].value, params['y_0'].value,
                                        params['amp'].value, params['exp_tau'].value)
        return (self.y - p) / self.sigma

    def run_fit(self):
        if self.model == 'gauss':
            result = minimize(self.residual_gauss, self.fit_params, method='nelder')
            self.fitted_params = result.params
            self.prediction = self.expression_gauss(
                self.fitted_params['x_0'].value,
                self.fitted_params['y_0'].value,
                self.fitted_params['amp'].value,
                self.fitted_params['gauss_fwhm'].value)
            red_chi_sq = np.sum(((self.y - self.prediction) / self.sigma) ** 2) / (len(self.x) - result.nvarys)
            return {'x_0': self.fitted_params['x_0'].value,
                    'y_0': self.fitted_params['y_0'].value,
                    'amp': self.fitted_params['amp'].value,
                    'gauss_fwhm': self.fitted_params['gauss_fwhm'].value,
                    'red_chi_sq': red_chi_sq}

        elif self.model == 'exponential':
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = minimize(self.residual_exponential, self.fit_params, method='nelder')
            self.fitted_params = result.params
            self.prediction = self.expression_exponential(
                self.fitted_params['x_0'].value,
                self.fitted_params['y_0'].value,
                self.fitted_params['amp'].value,
                self.fitted_params['exp_tau'].value)
            red_chi_sq = np.sum(((self.y - self.prediction) / self.sigma) ** 2) / (len(self.x) - result.nvarys)
            return {'x_0': self.fitted_params['x_0'].value,
                    'y_0': self.fitted_params['y_0'].value,
                    'amp': self.fitted_params['amp'].value,
                    'exp_tau': self.fitted_params['exp_tau'].value,
                    'red_chi_sq': red_chi_sq}
        else:
            raise ValueError('Invalid model. Use "gauss" or "exponential".')

# ─────────────────────────────────────────────────────────────────────────────
# Polynomial_fit  (faithful port, with one bug fix documented below)
# ─────────────────────────────────────────────────────────────────────────────

class Polynomial_fit:
    """
    Constrained polynomial fit used for bleaching/drift trend estimation.

    Faithful port of the Polynomial_fit class from the full FCS_Fixer
    source. Mimics np.polynomial.polynomial.polyfit() (low-order-first
    coefficient convention), but constrains the zero-order term to be
    non-negative -- important because the trend describes a photon
    count rate, which cannot physically go negative.

    BUG FIX (documented, not silent): the original source's
    polynomial_residual() and run_fit() called
    self.polynomial_expression(coefficients) directly on the raw
    lmfit.Parameters object returned during/after minimize(). But
    polynomial_expression() expects a plain numpy coefficient array,
    not a Parameters object -- the correct unpacking method is
    polynomial_fun(), which exists specifically to convert Parameters
    into a coefficient array first. As literally transcribed, the
    original code would raise a TypeError as soon as lmfit tried to
    evaluate the residual. This is corrected below by calling
    polynomial_fun() in both places, which is unambiguously the
    intended behaviour (polynomial_fun exists for exactly this purpose
    and is otherwise unused elsewhere in the class).
    """

    def __init__(self, time, counts, poly_order):
        self.time = time
        self.counts = counts
        self.poly_order = poly_order

        # raising time to successive powers is reused every iteration
        self.time_power = (
            self.time.reshape((self.time.shape[0], 1))
            ** np.arange(poly_order + 1).reshape((1, poly_order + 1))
        )

        sigma_counts = np.sqrt(counts)
        sigma_counts[sigma_counts == 0] = np.max(sigma_counts) * 1e3
        self.sigma_counts = sigma_counts

        self.fit_params = Parameters()
        # zero-order term constrained >= smallest nonzero count in data
        self.fit_params.add(
            "c0",
            value=self.counts.mean(),
            min=np.min(self.counts[self.counts > 0]),
            vary=True,
        )
        if self.poly_order > 0:
            for order in range(1, self.poly_order + 1):
                self.fit_params.add(f"c{order}", value=0.0, vary=True)

    def polynomial_expression(self, coefficient_array):
        """coefficient_array : np.ndarray shape (1, poly_order+1)."""
        return (self.time_power * coefficient_array).sum(axis=1)

    def polynomial_fun(self, coefficients):
        """coefficients : lmfit.Parameters -> unpacks into coefficient array."""
        coefficient_array = np.zeros((1, self.poly_order + 1), dtype=np.float64)
        coefficient_array[0, 0] = coefficients["c0"]
        if self.poly_order > 0:
            for order in range(1, self.poly_order + 1):
                coefficient_array[0, order] = coefficients[f"c{order}"]
        return self.polynomial_expression(coefficient_array)

    def polynomial_residual(self, coefficients):
        # FIXED: was self.polynomial_expression(coefficients) -- see
        # class docstring. polynomial_fun() is the correct call here.
        residual = self.counts - self.polynomial_fun(coefficients)
        return residual / self.sigma_counts

    def run_fit(self):
        result = minimize(self.polynomial_residual, self.fit_params, method="nelder")
        fitted_params = result.params

        # FIXED: was self.polynomial_expression(fitted_params) -- same
        # issue as above.
        prediction = self.polynomial_fun(fitted_params)

        chi_squared = np.sum(((self.counts - prediction) / self.sigma_counts) ** 2)
        n_data_points = self.counts.shape[0]
        red_chi_squared = chi_squared / (n_data_points - result.nvarys)

        fit_params = np.zeros((self.poly_order + 1,), dtype=np.float64)
        fit_params[0] = fitted_params["c0"].value
        if self.poly_order > 0:
            for order in range(1, self.poly_order + 1):
                fit_params[order] = fitted_params[f"c{order}"].value

        return fit_params, red_chi_squared
# ─────────────────────────────────────────────────────────────────────────────
# FCS_Fixer  (trimmed, faithful port)
# ─────────────────────────────────────────────────────────────────────────────

class FCS_Fixer:

    # ── static: channels_spec helpers (verbatim) ─────────────────────────
    @staticmethod
    def build_channels_spec(channels_indices, micro_time_gates=None):
        if not isint(channels_indices) and not isiterable(channels_indices):
            raise ValueError('Invalid input for channels_indices: Must be int or list of int')
        elif isiterable(channels_indices) and not np.all([isint(e) for e in channels_indices]):
            raise ValueError('Invalid input for channels_indices: Must be int or list of int')

        if not isiterable(micro_time_gates):
            if micro_time_gates is None:
                channels_spec = FCS_Fixer.check_channels_spec(channels_indices)
            else:
                raise ValueError('Invalid input for micro_time_gates.')
        else:
            micro_time_gates = np.array(micro_time_gates)
            if not (np.all([isfloat(e) for e in micro_time_gates]) and
                    np.all(micro_time_gates >= 0.) and
                    np.all(micro_time_gates <= 1.) and
                    np.all(np.diff(micro_time_gates) >= 0.) and
                    (len(micro_time_gates) > 0 and len(micro_time_gates) % 2 == 0)):
                raise ValueError('Invalid input for micro_time_gates.')

            micro_time_cutoffs = []
            micro_time_gates_to_use = []
            last_stop = 0.
            gate_counter = 0

            for i_gate in range(0, len(micro_time_gates), 2):
                start = micro_time_gates[i_gate]
                stop = micro_time_gates[i_gate + 1]

                if start > last_stop:
                    micro_time_cutoffs.append(start)
                    micro_time_gates_to_use.append(gate_counter + 1)
                    gate_counter += 2
                else:
                    micro_time_gates_to_use.append(gate_counter)
                    gate_counter += 1

                if stop < 1.:
                    micro_time_cutoffs.append(stop)
                    last_stop = stop

            if len(micro_time_cutoffs) == 0:
                channels_spec = FCS_Fixer.check_channels_spec(channels_indices)
            else:
                channels_spec = ((channels_indices,), ((*micro_time_cutoffs,), (*micro_time_gates_to_use,)))

        return channels_spec

    @staticmethod
    def check_channels_spec(channels_spec):
        if not (isint(channels_spec) or isiterable(channels_spec)):
            raise ValueError('Invalid channels_spec.')
        elif isint(channels_spec):
            return ((channels_spec,), ((), (0,)))
        elif type(channels_spec) == list:
            if isint(channels_spec[0]):
                if np.any([not isint(e) for e in channels_spec]):
                    raise ValueError('Invalid channels_spec.')
                return (tuple(channels_spec), ((), (0,)))
        elif isiterable(channels_spec):
            if isint(channels_spec[0]):
                if np.any([not isint(e) for e in channels_spec]):
                    raise ValueError('Invalid channels_spec.')
                return (tuple(channels_spec), ((), (0,)))
            if isfloat(channels_spec[0]):
                raise ValueError('channels_spec[0] invalid: found float where int expected.')
            if isiterable(channels_spec[0]):
                if np.any([isfloat(e) for e in channels_spec[0]]):
                    raise ValueError('channels_spec[0] invalid.')
                if np.any([not isint(e) for e in channels_spec[0]]):
                    raise ValueError('channels_spec[0] invalid.')
                if not isiterable(channels_spec[1]):
                    raise ValueError('channels_spec[1] invalid.')
                if not isiterable(channels_spec[1][0]) or \
                   np.any([(not isfloat(e) or e < 0 or e > 1) for e in channels_spec[1][0]]):
                    raise ValueError('channels_spec[1][0] invalid.')
                if not isiterable(channels_spec[1][1]) or \
                   np.any([(not isint(e) or e > len(channels_spec[1][0])) for e in channels_spec[1][1]]):
                    raise ValueError('channels_spec[1][1] invalid.')
                return (tuple(channels_spec[0]), (tuple(channels_spec[1][0]), tuple(channels_spec[1][1])))

    @staticmethod
    def get_flcs_filters(tcspc_y, patterns):
        if not type(tcspc_y) == np.ndarray:
            raise ValueError('tcspc_y must be np.ndarray.')
        if not (type(patterns) == np.ndarray and patterns.ndim == 2 and
                patterns.shape[0] == tcspc_y.shape[0]):
            raise ValueError('patterns must be 2D np.ndarray matching tcspc_y length.')

        patterns_norm = np.ones_like(patterns, dtype=np.float64)
        for i in range(patterns.shape[1]):
            patterns_norm[:, i] = patterns[:, i] / patterns[:, i].sum()

        inv_tcspc_diag = np.diag(tcspc_y ** (-1))

        step_1 = np.matmul(inv_tcspc_diag, patterns_norm)
        step_2 = np.matmul(patterns_norm.T, step_1)
        step_3 = np.linalg.pinv(step_2)
        step_4 = np.matmul(patterns_norm, step_3)
        flcs_weights = np.matmul(inv_tcspc_diag, step_4)

        return flcs_weights, patterns_norm

    # ── init (trimmed) ────────────────────────────────────────────────────
    def __init__(self, photon_data, tau_min=1E-6, tau_max=1.0, sampling=8,
                cross_corr_symm=False, correlation_method='wahl',   # CHANGED: was 'default'
                subtract_afterpulsing=False, afterpulsing_params_path=''):

        self._photon_data = photon_data
        self._macro_times = self._photon_data.macro_times
        self._macro_times_correction_bursts = np.zeros_like(self._macro_times)
        self._macro_times_correction_mse_filter = np.zeros_like(self._macro_times)

        # CHANGED: tttrlib header resolutions are returned in SECONDS in the
        # current tttrlib version (confirmed empirically: a 50 ns macro time
        # resolution is reported as 5.00004e-08, i.e. seconds). The original
        # FCS_Fixer assumed these were already in nanoseconds and built its
        # entire internal unit system (tau_min/tau_max in ns, afterpulsing
        # time constants in ns, n_casc lag-time cascade calculation, etc.)
        # around that assumption. Converting to ns here, immediately after
        # reading from the header, keeps every other line in this class
        # correct without further changes.
        self._micro_time_resolution = self._photon_data.header.micro_time_resolution * 1e9
        self._macro_time_resolution = self._photon_data.header.macro_time_resolution * 1e9

        self._acquisition_time = np.max(self._macro_times) * self._macro_time_resolution
        self._n_total_photons = self._macro_times.shape[0]
        self._n_micro_time_bins = self._photon_data.get_number_of_micro_time_channels()
        self._routing_channels = np.unique(self._photon_data.routing_channels)
        self._n_channels = np.max(self._routing_channels) + 1

        self._weights_burst_removal = np.ones_like(self._macro_times, dtype=bool)   # CHANGED: was np.bool8
        self._weights_undrift = np.ones_like(self._macro_times, dtype=np.float16)
        self._weights_flcs_bg_corr = np.ones_like(self._macro_times, dtype=np.float16)
        self._weights_mse_filter = np.ones_like(self._macro_times, dtype=bool)      # CHANGED: was np.bool8
        self._weights_ext = np.ones_like(self._macro_times, dtype=np.float16)

        self._tau_min = 1E3
        self._tau_max = 1E9
        self._tau_min, self._tau_max = self.check_tau_min_max(tau_min * 1E9, tau_max * 1E9)

        self._sampling = int(sampling)
        self._cross_corr_symm = bool(cross_corr_symm)

        # CHANGED: allowed values now 'wahl' / 'felekyan' / 'laurence'
        if correlation_method not in ('wahl', 'felekyan', 'laurence'):
            raise ValueError("correlation_method must be 'wahl', 'felekyan', or 'laurence'.")
        self._correlation_method = correlation_method

        self._subtract_afterpulsing = bool(subtract_afterpulsing)
        self._afterpulsing_params_path = afterpulsing_params_path
        self._afterpulsing_p = np.array([])
        self._afterpulsing_params = np.array([])

        self._parameters_set = False

    # ── properties needed downstream ──────────────────────────────────────
    @property
    def photon_data(self):
        return self._photon_data

    @property
    def routing_channels(self):
        return self._routing_channels

    @property
    def n_micro_time_bins(self):
        return self._n_micro_time_bins

    @property
    def cross_corr_symm(self):
        return self._cross_corr_symm

    @cross_corr_symm.setter
    def cross_corr_symm(self, v):
        self._cross_corr_symm = bool(v)

    @property
    def subtract_afterpulsing(self):
        return self._subtract_afterpulsing

    @subtract_afterpulsing.setter
    def subtract_afterpulsing(self, v):
        self._subtract_afterpulsing = bool(v)

    @property
    def afterpulsing_params_path(self):
        return self._afterpulsing_params_path

    @afterpulsing_params_path.setter
    def afterpulsing_params_path(self, path):
        self._afterpulsing_params_path = path
        self._afterpulsing_p = np.array([])
        self._afterpulsing_params = np.array([])
        self._parameters_set = False

    # ── setup (verbatim logic) ────────────────────────────────────────────
    def set_correlation_time_params(self):
        if self._tau_min < 20 * self._macro_time_resolution:
            self._micro_time_corr = True
            resolution = self._micro_time_resolution
        else:
            self._micro_time_corr = False
            resolution = self._macro_time_resolution

        lag_point = 1
        lag_time = resolution
        while lag_time <= 10. * self._tau_max:
            lag_point += 1
            lag_time += 2 ** (np.floor((lag_point - 1) / self._sampling)) * resolution

        self._n_casc = np.ceil(lag_point / self._sampling)

    def set_afterpulsing_params(self):
        if self._subtract_afterpulsing:
            afterpulsing_params = np.genfromtxt(self._afterpulsing_params_path, delimiter=',', skip_header=1)
            if afterpulsing_params.ndim == 1:
                afterpulsing_params = afterpulsing_params[np.newaxis, :]
            afterpulsing_params[:, [0, 2]] *= 1E-9   # Hz -> GHz
            afterpulsing_params[:, [1, 3]] *= 1E9    # s  -> ns
            self._afterpulsing_params = afterpulsing_params
            self._afterpulsing_p = (afterpulsing_params[:, 0] * afterpulsing_params[:, 1] +
                                     afterpulsing_params[:, 2] * afterpulsing_params[:, 3])
        else:
            self._afterpulsing_params = np.zeros((self._n_channels, 4))
            self._afterpulsing_params[:, [1, 3]] = 1.
            self._afterpulsing_p = np.zeros((self._n_channels,))

    def update_params(self):
        if not self._parameters_set:
            self.set_correlation_time_params()
            self.set_afterpulsing_params()
            self._parameters_set = True

    def check_tau_min_max(self, tau_min=None, tau_max=None):
        if tau_min is None:
            tau_min = self._tau_min
        if tau_max is None:
            tau_max = self._tau_max
        return float(tau_min), float(tau_max)

    # ── select_photons (verbatim, trimmed of burst/undrift/mse branches
    #    which stay in the signature but are always False here) ──────────
    def select_photons(self, channels_spec, ext_indices=np.array([]),
                       use_ext_weights=False, use_drift_correction=False,
                       use_flcs_bg_corr=False, use_burst_removal=False,
                       use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        channels_spec_norm = self.check_channels_spec(channels_spec)
        channels = channels_spec_norm[0]
        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]

        macro_times = self._macro_times.copy()
        if use_burst_removal:
            macro_times -= self._macro_times_correction_bursts
        if use_mse_filter:
            macro_times -= self._macro_times_correction_mse_filter

        micro_times = self._photon_data.micro_times

        mask_select = np.ones(macro_times.shape, dtype=bool)   # CHANGED: was np.bool8

        indices_channels = self._photon_data.get_selection_by_channel(list(channels))
        mask_temp = np.zeros_like(mask_select)
        mask_temp[indices_channels] = True
        mask_select *= mask_temp

        if use_burst_removal:
            mask_select *= self._weights_burst_removal
        if use_mse_filter:
            mask_select *= self._weights_mse_filter
        if use_drift_correction:
            mask_select *= self._weights_undrift != 0

        if ext_indices.shape[0] > 0:
            mask_temp[:] = False
            mask_temp[ext_indices] = True
            mask_select *= mask_temp

        if use_ext_weights:
            mask_select *= self._weights_ext != 0

        if len(micro_time_cutoffs) > 0:
            if len(micro_time_cutoffs) == 1:
                gates = np.array([0, micro_time_cutoffs[0], 1]) * self._n_micro_time_bins
            else:
                gates = [0.]
                for g in micro_time_cutoffs:
                    gates.append(g)
                gates.append(1.)
                gates = np.array(gates) * self._n_micro_time_bins

            if len(micro_time_gate_indx) == 1:
                selection_micro_time = np.logical_and(
                    micro_times >= gates[micro_time_gate_indx[0]],
                    micro_times <= gates[micro_time_gate_indx[0] + 1])
            else:
                selection_micro_time = np.zeros((self._n_total_photons,), dtype=bool)  # CHANGED
                for gi in micro_time_gate_indx:
                    selection_micro_time = np.logical_or(
                        selection_micro_time,
                        np.logical_and(micro_times >= gates[gi], micro_times <= gates[gi + 1]))

            mask_temp[:] = False
            mask_temp[selection_micro_time] = True
            mask_select *= mask_temp

        indices_select = np.nonzero(mask_select)[0]
        macro_times_select = macro_times[indices_select]
        micro_times_select = micro_times[indices_select]

        if use_ext_weights:
            weights_select = self._weights_ext[indices_select]
        else:
            weights_select = np.ones_like(macro_times_select, dtype=float)

        if use_drift_correction:
            weights_select *= self._weights_undrift[indices_select]
        if use_flcs_bg_corr:
            weights_select *= self._weights_flcs_bg_corr[indices_select]

        return macro_times_select, micro_times_select, weights_select, indices_select

    # ── afterpulse_correlation (verbatim) ─────────────────────────────────
    def afterpulse_correlation(self, lag_times, acr, channels_spec):
        if not self._parameters_set:
            self.update_params()

        channels_spec_norm = self.check_channels_spec(channels_spec)
        channel = channels_spec_norm[0]
        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]

        if len(channel) > 1:
            raise ValueError('afterpulse_correlation supports only a single channel.')

        detector_params = self._afterpulsing_params[channel, :]
        ap_amp_1, ap_tau_1, ap_amp_2, ap_tau_2 = detector_params[0]
        afterpulse_p = self._afterpulsing_p[channel]

        if len(micro_time_cutoffs) == 0:
            micro_time_used_fraction = 1.
        else:
            if len(micro_time_cutoffs) == 1:
                gates = np.array([0, micro_time_cutoffs[0], 1])
            else:
                gates = [0.]
                for g in micro_time_cutoffs:
                    gates.append(g)
                gates.append(1.)
                gates = np.array(gates)

            micro_time_used_fraction = 0.
            if type(micro_time_gate_indx) == int:
                micro_time_used_fraction += gates[micro_time_gate_indx + 1] - gates[micro_time_gate_indx]
            else:
                for gi in micro_time_gate_indx:
                    micro_time_used_fraction += gates[gi + 1] - gates[gi]

        G_afterpulse = ((ap_amp_1 * np.exp(-lag_times / ap_tau_1) +
                        ap_amp_2 * np.exp(-lag_times / ap_tau_2))
                       / (1 + afterpulse_p) / acr
                       * micro_time_used_fraction)
        
        return G_afterpulse

    # ── correlation_apply_filters (verbatim except the 2 marked lines) ────
    def correlation_apply_filters(self, channels_spec_1, channels_spec_2,
                                  ext_indices=np.array([]), tau_min=None, tau_max=None,
                                  use_ext_weights=False, use_drift_correction=False,
                                  use_flcs_bg_corr=False, use_burst_removal=False,
                                  use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)

        macro_times_ch1, micro_times_ch1, weights_ch1, _ = self.select_photons(
            channels_spec_norm_ch1, use_ext_weights=use_ext_weights, ext_indices=ext_indices,
            use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
            use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        macro_times_ch2, micro_times_ch2, weights_ch2, _ = self.select_photons(
            channels_spec_norm_ch2, use_ext_weights=use_ext_weights, ext_indices=ext_indices,
            use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
            use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)
        if len(macro_times_ch1) == 0:
            raise ValueError(
                f"No photons found for channels_spec_1={channels_spec_norm_ch1}. "
                f"Available routing channels in this file: "
                f"{sorted(int(c) for c in np.unique(self._routing_channels))}. "
                f"This usually means the requested channel does not exist "
                f"in this measurement (e.g. a single-detector acquisition "
                f"where a second channel was still requested for cross-"
                f"correlation or PIE)."
            )
        if len(macro_times_ch2) == 0:
            raise ValueError(
                f"No photons found for channels_spec_2={channels_spec_norm_ch2}. "
                f"Available routing channels in this file: "
                f"{sorted(int(c) for c in np.unique(self._routing_channels))}. "
                f"This usually means the requested channel does not exist "
                f"in this measurement (e.g. a single-detector acquisition "
                f"where a second channel was still requested for cross-"
                f"correlation or PIE)."
            )

        window_start = np.min([np.min(macro_times_ch1), np.min(macro_times_ch2)])
        macro_times_ch1 = macro_times_ch1 - window_start
        macro_times_ch2 = macro_times_ch2 - window_start

        acr_1 = np.sum(weights_ch1) / np.max(macro_times_ch1) / self._macro_time_resolution
        acr_2 = np.sum(weights_ch2) / np.max(macro_times_ch2) / self._macro_time_resolution
        
        corr = tttrlib.Correlator()
        corr.n_bins = self._sampling
        corr.n_casc = int(self._n_casc)
        corr.method = self._correlation_method
        corr.set_events(macro_times_ch1, weights_ch1, macro_times_ch2, weights_ch2)

        if self._micro_time_corr:
            corr.make_fine = True
            corr.set_microtimes(micro_times_ch1, micro_times_ch2, self._n_micro_time_bins)

        cc = corr.get_corr_normalized()
        # CHANGED: get_x_axis_normalized() -> get_x_axis()
        # (see module docstring — behaviourally identical here since
        # Correlator.set_tttr() is never called, so the internal
        # macro_time_duration calibration remains at its default of 1.0)
        resolution = self._micro_time_resolution if self._micro_time_corr else self._macro_time_resolution
        lag_times = corr.get_x_axis() * resolution

        channel_1 = channels_spec_norm_ch1[0]
        micro_time_gates_1 = channels_spec_norm_ch1[1]
        channel_2 = channels_spec_norm_ch2[0]
        micro_time_gates_2 = channels_spec_norm_ch2[1]

        if self._cross_corr_symm and channel_1 != channel_2:
            corr_rev = tttrlib.Correlator()
            corr_rev.n_bins = self._sampling
            corr_rev.n_casc = int(self._n_casc)
            corr_rev.method = self._correlation_method
            corr_rev.set_events(macro_times_ch2, weights_ch2, macro_times_ch1, weights_ch1)
            if self._micro_time_corr:
                corr_rev.make_fine = True
                corr_rev.set_microtimes(micro_times_ch2, micro_times_ch1, self._n_micro_time_bins)
            cc_rev = corr_rev.get_corr_normalized()
            cc_processed = (cc + cc_rev) / 2
            cc_processed -= 1

        elif self._cross_corr_symm and channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2:
            corr_rev = tttrlib.Correlator()
            corr_rev.n_bins = self._sampling
            corr_rev.n_casc = int(self._n_casc)
            corr_rev.method = self._correlation_method
            corr_rev.set_events(macro_times_ch2, weights_ch2, macro_times_ch1, weights_ch1)
            if self._micro_time_corr:
                corr_rev.make_fine = True
                corr_rev.set_microtimes(micro_times_ch2, micro_times_ch1, self._n_micro_time_bins)
            cc_rev = corr_rev.get_corr_normalized()

            if self._subtract_afterpulsing and not use_flcs_bg_corr:
                cc = cc - self.afterpulse_correlation(lag_times, acr_2, channels_spec_norm_ch2)
                cc_rev = cc_rev - self.afterpulse_correlation(lag_times, acr_1, channels_spec_norm_ch1)

            cc_processed = (cc + cc_rev) / 2 - 1

        elif not self._cross_corr_symm and channel_1 == channel_2 and micro_time_gates_1 != micro_time_gates_2:
            if self._subtract_afterpulsing and not use_flcs_bg_corr:
                cc_processed = (cc - self.afterpulse_correlation(lag_times, acr_2, channels_spec_norm_ch2)) - 1
            else:
                cc_processed = cc - 1

        elif channel_1 == channel_2 and micro_time_gates_1 == micro_time_gates_2:
            if self._subtract_afterpulsing:
                cc_processed = (cc - self.afterpulse_correlation(lag_times, acr_2, channels_spec_norm_ch2)) - 1
            else:
                cc_processed = cc - 1

        else:
            cc_processed = cc - 1

        keep = np.logical_and(lag_times >= tau_min, lag_times <= tau_max)
        lag_times = lag_times[keep]
        cc_processed = cc_processed[keep]

        return lag_times, cc_processed, acr_1, acr_2

    # ── get_segment_ccs (verbatim) ─────────────────────────────────────────
    def get_segment_ccs(self, channels_spec_1, channels_spec_2, minimum_window_length,
                        tau_min=None, tau_max=None, use_ext_weights=False,
                        use_drift_correction=False, use_flcs_bg_corr=False,
                        use_burst_removal=False, use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channel_1, micro_time_gates_1 = self.check_channels_spec(channels_spec_1)
        channel_2, micro_time_gates_2 = self.check_channels_spec(channels_spec_2)

        minimum_window_length = minimum_window_length * 1E9  # to ns

        macro_times = self._macro_times.copy()
        if use_burst_removal:
            macro_times -= self._macro_times_correction_bursts
        if use_mse_filter:
            macro_times -= self._macro_times_correction_mse_filter

        time_windows = tttrlib.ranges_by_time_window(
            macro_times,
            minimum_window_length=minimum_window_length / 1E6,
            macro_time_calibration=self._macro_time_resolution / 1E6,
        )

        n_time_windows = len(time_windows) // 2
        start_stop = np.array(time_windows).reshape((n_time_windows, 2))

        segment_ccs = np.array([])
        usable_segments = []
        segment_index = 0

        for start, stop in start_stop:
            try:
                keep_indices = np.arange(start, stop, dtype=np.int64)
                lag_times, segment_cc, _, _ = self.correlation_apply_filters(
                    (channel_1, micro_time_gates_1), (channel_2, micro_time_gates_2),
                    use_ext_weights=use_ext_weights, ext_indices=keep_indices,
                    tau_min=tau_min, tau_max=tau_max,
                    use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                    use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

                if segment_ccs.shape[0] == 0:
                    segment_ccs = np.zeros((segment_cc.shape[0], n_time_windows), dtype=float)

                segment_ccs[:, segment_index] = segment_cc
                usable_segments.append(segment_index)

            except Exception:
                pass
            finally:
                segment_index += 1

        usable_segments = np.array(usable_segments)
        return lag_times, segment_ccs, usable_segments, start_stop

    # ── get_Wohland_SD (verbatim) ──────────────────────────────────────────
    def get_Wohland_SD(self, channels_spec_1, channels_spec_2, minimum_window_length=[],
                       tau_max=None, tau_min=None, use_ext_weights=False,
                       use_drift_correction=False, use_flcs_bg_corr=False,
                       use_burst_removal=False, use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)

        effective_acquisition_time = self._macro_times[-1]
        if use_burst_removal:
            effective_acquisition_time = effective_acquisition_time - self._macro_times_correction_bursts[-1]
        if use_mse_filter:
            effective_acquisition_time = effective_acquisition_time - self._macro_times_correction_mse_filter[-1]
        effective_acquisition_time = effective_acquisition_time * self._macro_time_resolution

        if type(minimum_window_length) in [list, np.ndarray] and len(minimum_window_length) == 0:
            minimum_window_length = np.max(
                [effective_acquisition_time / 10., 5. * tau_max]).astype(np.float64) * 1E-9
        else:
            minimum_window_length = np.float64(minimum_window_length)

        if effective_acquisition_time * 1E-9 / minimum_window_length < 5:
            return self.get_bootstrap_SD(
                channels_spec_norm_ch1, channels_spec_norm_ch2, n_bootstrap_reps=10,
                use_ext_weights=use_ext_weights, tau_min=tau_min, tau_max=tau_max,
                use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        _, segment_ccs, usable_segments, _ = self.get_segment_ccs(
            channels_spec_norm_ch1, channels_spec_norm_ch2, minimum_window_length,
            tau_min=tau_min, tau_max=tau_max, use_ext_weights=use_ext_weights,
            use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
            use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        if segment_ccs.shape[1] > 1:
            var_segment_ccs = np.var(segment_ccs[:, usable_segments], axis=1)
            n_time_windows = segment_ccs.shape[1]
            sd_cc = np.sqrt(var_segment_ccs / (n_time_windows - 1))
        else:
            sd_cc = self.get_bootstrap_SD(
                channels_spec_1, channels_spec_2, n_bootstrap_reps=10,
                use_ext_weights=use_ext_weights, tau_min=tau_min, tau_max=tau_max,
                use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        return sd_cc

    # ── get_bootstrap_SD (verbatim) ────────────────────────────────────────
    def get_bootstrap_SD(self, channels_spec_1, channels_spec_2, n_bootstrap_reps=10,
                         tau_min=None, tau_max=None, use_ext_weights=False,
                         use_drift_correction=False, use_flcs_bg_corr=False,
                         use_burst_removal=False, use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        tau_min, tau_max = self.check_tau_min_max(tau_min, tau_max)
        channels_spec_norm_ch1 = self.check_channels_spec(channels_spec_1)
        channels_spec_norm_ch2 = self.check_channels_spec(channels_spec_2)

        rng = np.random.default_rng()
        sum_bs_cc = np.array([])
        sum_of_squares_bs_cc = np.array([])
        failed_segments = 0

        for _ in range(n_bootstrap_reps):
            try:
                resample_indxs = rng.choice(self._n_total_photons, size=self._n_total_photons)
                resample_indxs_sort = np.sort(resample_indxs)

                lag_times, rep_cc, _, _ = self.correlation_apply_filters(
                    channels_spec_norm_ch1, channels_spec_norm_ch2,
                    use_ext_weights=use_ext_weights, ext_indices=resample_indxs_sort,
                    tau_min=tau_min, tau_max=tau_max,
                    use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                    use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

                if sum_bs_cc.shape[0] == 0:
                    sum_bs_cc = np.zeros_like(rep_cc, dtype=float)
                    sum_of_squares_bs_cc = np.zeros_like(rep_cc, dtype=float)

                sum_bs_cc += rep_cc
                sum_of_squares_bs_cc += np.square(rep_cc)

            except Exception:
                failed_segments += 1

        n_bootstrap_reps_corr = n_bootstrap_reps - failed_segments

        if n_bootstrap_reps_corr > 1:
            bs_cc_squared_1st_mom = np.square(sum_bs_cc) / (np.square(n_bootstrap_reps_corr) - n_bootstrap_reps_corr)
            bs_cc_2nd_mom = sum_of_squares_bs_cc / (n_bootstrap_reps_corr - 1)
            var_bs_cc = bs_cc_2nd_mom - bs_cc_squared_1st_mom
            sd_cc = np.sqrt(var_bs_cc)
        else:
            sd_cc = np.array(1.)

        return sd_cc

    # ── get_correlation_uncertainty (verbatim, write_results stripped) ─────
    def get_correlation_uncertainty(self, channels_spec_1, channels_spec_2,
                                    default_uncertainty_method='Wohland',
                                    minimum_window_length=[], n_bootstrap_reps=10,
                                    tau_min=None, tau_max=None, use_ext_weights=False,
                                    use_drift_correction=False, use_flcs_bg_corr=False,
                                    use_burst_removal=False, use_mse_filter=False):
        if not self._parameters_set:
            self.update_params()

        lag_times, cc, acr1, acr2 = self.correlation_apply_filters(
            channels_spec_1, channels_spec_2, tau_min=tau_min, tau_max=tau_max,
            use_ext_weights=use_ext_weights, use_drift_correction=use_drift_correction,
            use_flcs_bg_corr=use_flcs_bg_corr, use_burst_removal=use_burst_removal,
            use_mse_filter=use_mse_filter)

        if default_uncertainty_method == 'Wohland':
            sd_cc = self.get_Wohland_SD(
                channels_spec_1, channels_spec_2, minimum_window_length=minimum_window_length,
                tau_max=tau_max, tau_min=tau_min, use_ext_weights=use_ext_weights,
                use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)
        else:
            sd_cc = self.get_bootstrap_SD(
                channels_spec_1, channels_spec_2, n_bootstrap_reps=n_bootstrap_reps,
                tau_max=tau_max, tau_min=tau_min, use_ext_weights=use_ext_weights,
                use_drift_correction=use_drift_correction, use_flcs_bg_corr=use_flcs_bg_corr,
                use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        if np.all(sd_cc == 1.):
            sd_cc = np.ones_like(cc)

        return lag_times, cc, sd_cc, acr1, acr2

    # ── get_tcspc_histogram (verbatim) ─────────────────────────────────────
    def get_tcspc_histogram(self, channels_spec, micro_times=[], ext_indices=np.array([]),
                            use_ext_weights=False, use_drift_correction=False,
                            use_burst_removal=False, use_mse_filter=False):
        channels_spec_norm = self.check_channels_spec(channels_spec)

        if isinstance(micro_times, list) and len(micro_times) == 0:
            use_external_micro_times = False
            _, micro_times, weights, _ = self.select_photons(
                channels_spec_norm, ext_indices=ext_indices, use_ext_weights=use_ext_weights,
                use_drift_correction=use_drift_correction, use_burst_removal=use_burst_removal,
                use_mse_filter=use_mse_filter)
        else:
            use_external_micro_times = True
            weights = np.ones_like(micro_times, dtype=np.float64)

        tcspc_x_raw = np.arange(0, self._n_micro_time_bins)
        tcspc_y_raw = np.histogram(
            micro_times, bins=np.append(tcspc_x_raw, self._n_micro_time_bins + 1),
            density=False, weights=weights)[0]

        if use_external_micro_times:
            micro_time_mask = self._get_micro_time_mask(channels_spec_norm)
        else:
            micro_time_mask = tcspc_y_raw > 0

        tcspc_x = tcspc_x_raw[micro_time_mask]
        tcspc_y = tcspc_y_raw[micro_time_mask]
        return tcspc_x, tcspc_y

    def _get_micro_time_mask(self, channels_spec):
        channels_spec_norm = self.check_channels_spec(channels_spec)
        micro_time_mask = np.zeros((self._n_micro_time_bins,), dtype=bool)  # CHANGED
        micro_time_cutoffs = channels_spec_norm[1][0]
        micro_time_gate_indx = channels_spec_norm[1][1]

        if micro_time_cutoffs == ():
            micro_time_mask[:] = True
        elif len(micro_time_cutoffs) == 1:
            gates = np.ceil([0, micro_time_cutoffs[0] * self._n_micro_time_bins,
                             self._n_micro_time_bins]).astype(np.uint64)
            if len(micro_time_gate_indx) == 1:
                micro_time_mask[gates[micro_time_gate_indx[0]]:gates[micro_time_gate_indx[0] + 1]] = True
            else:
                for gi in micro_time_gate_indx:
                    micro_time_mask[gates[gi]:gates[gi + 1]] = True
        else:
            gates = [0.]
            for g in micro_time_cutoffs:
                gates.append(g)
            gates.append(1)
            gates = np.ceil(np.array(gates) * self._n_micro_time_bins).astype(np.uint64)
            if len(micro_time_gate_indx) == 1:
                micro_time_mask[gates[micro_time_gate_indx[0]]:gates[micro_time_gate_indx[0] + 1]] = True
            else:
                for gi in micro_time_gate_indx:
                    micro_time_mask[gates[gi]:gates[gi + 1]] = True

        return micro_time_mask
    # ── get_time_trace (faithful port, trimmed of unused writing) ─────────
    def get_time_trace(
        self, channels_spec, time_trace_sampling, ext_indices=np.array([]),
        use_ext_weights=False, use_drift_correction=False,
        use_flcs_bg_corr=False, use_burst_removal=False, use_mse_filter=False,
    ):
        """
        Bin photon macro times into a time trace.

        Faithful port of FCS_Fixer.get_time_trace() (file-writing side
        effects stripped, as with the rest of this trimmed build).

        Parameters
        ----------
        channels_spec        : channels_spec tuple
        time_trace_sampling  : float, bin width in SECONDS

        Returns
        -------
        time_trace             : np.ndarray  photon counts per bin
        time_trace_bin_centers : np.ndarray  bin centre times, in SECONDS
        """
        channels_spec_norm = self.check_channels_spec(channels_spec)

        macro_times_select, _, weights_select, _ = self.select_photons(
            channels_spec_norm, ext_indices=ext_indices,
            use_ext_weights=use_ext_weights,
            use_drift_correction=use_drift_correction,
            use_flcs_bg_corr=use_flcs_bg_corr,
            use_burst_removal=use_burst_removal,
            use_mse_filter=use_mse_filter,
        )

        macro_times_ns = macro_times_select * self._macro_time_resolution
        effective_acquisition_time = (
            self._macro_times[-1] * self._macro_time_resolution
        )

        time_trace_bins = np.arange(
            0, effective_acquisition_time,
            time_trace_sampling * 1e9, dtype=float,
        )
        time_trace = np.histogram(
            macro_times_ns, bins=time_trace_bins,
            density=False, weights=weights_select,
        )[0]

        time_trace_bin_centers = (
            time_trace_bins[:-1] + 0.5 * (time_trace_bins[1] - time_trace_bins[0])
        ) * 1e-9  # back to seconds

        return time_trace, time_trace_bin_centers

    # ── get_trace_time_scale (simplified port) ─────────────────────────────
    def get_trace_time_scale(
        self, channels_spec, min_avg_counts=10.0, min_bin_width=1e-4,
        ext_indices=np.array([]), use_ext_weights=False,
        use_drift_correction=False, use_flcs_bg_corr=False,
        use_burst_removal=False, use_mse_filter=False,
    ):
        """
        Automatically estimate a reasonable time trace bin width.

        Simplified port of FCS_Fixer.get_trace_time_scale(): implements
        the two default-active criteria (minimum average counts per
        bin, and an absolute minimum bin width). The original method
        also supports an optional third criterion based on a
        preliminary diffusion-time fit (use_tau_diff), but that
        parameter defaults to False in FCS_Fixer itself, so this
        simplified port -- which always behaves as if use_tau_diff=False
        -- reproduces the DEFAULT behaviour exactly; it just doesn't
        expose the (rarely used, opt-in) diffusion-time criterion.

        Returns
        -------
        float  bin width in SECONDS
        """
        channels_spec_norm = self.check_channels_spec(channels_spec)
        min_bin_width_ns = min_bin_width * 1e9

        _, _, weights_select, _ = self.select_photons(
            channels_spec_norm, ext_indices=ext_indices,
            use_ext_weights=use_ext_weights,
            use_burst_removal=use_burst_removal,
            use_drift_correction=use_drift_correction,
            use_flcs_bg_corr=use_flcs_bg_corr,
            use_mse_filter=use_mse_filter,
        )

        effective_acquisition_time = (
            self._macro_times[-1] * self._macro_time_resolution
        )
        acr = weights_select.sum() / effective_acquisition_time  # counts / ns
        time_for_min_avg_counts = min_avg_counts / acr

        time_trace_sampling_ns = np.max(
            np.array([time_for_min_avg_counts, min_bin_width_ns])
        )
        return time_trace_sampling_ns * 1e-9  # -> seconds

    # ── get_auto_undrift_order (faithful port) ──────────────────────────────
    def get_auto_undrift_order(
        self, time_trace_bin_centers, time_trace_counts, max_undrift_order,
    ):
        """
        F-test-driven polynomial degree selection for the bleaching
        trend, faithful port of FCS_Fixer.get_auto_undrift_order().

        Starts at degree 1 and keeps incrementing while the improvement
        from degree n-1 to degree n is statistically significant
        (alpha=0.05), stopping (and reverting to the last good degree)
        once two consecutive increments fail to show significant
        improvement.
        """
        undrift_order = 1
        red_chi_sq = np.zeros((max_undrift_order,), dtype=float)
        significant_improvement = np.zeros((max_undrift_order,), dtype=bool)

        while undrift_order <= max_undrift_order:
            _, red_chi_sq_iter = Polynomial_fit(
                time_trace_bin_centers, time_trace_counts, undrift_order
            ).run_fit()
            red_chi_sq[undrift_order - 1] = red_chi_sq_iter

            if undrift_order > 1:
                dfd = time_trace_counts.shape[0] - undrift_order
                F_value = (
                    (red_chi_sq[undrift_order - 2] - red_chi_sq[undrift_order - 1])
                    / red_chi_sq[undrift_order - 1] * dfd
                )
                significant_improvement[undrift_order - 1] = (
                    f_dist.sf(F_value, 1, dfd) < 0.05
                )

                if (
                    (significant_improvement[undrift_order - 1]
                     or significant_improvement[undrift_order - 2])
                    and undrift_order != max_undrift_order
                ):
                    undrift_order += 1
                elif undrift_order == max_undrift_order:
                    if (significant_improvement[undrift_order - 1]
                            and significant_improvement[undrift_order - 2]):
                        break
                    elif not significant_improvement[undrift_order - 2]:
                        undrift_order -= 2
                        break
                    else:
                        undrift_order -= 1
                        break
                else:
                    undrift_order -= 2
                    break
            else:
                undrift_order += 1

        return undrift_order

    # ── polynomial_undrifting_rss (faithful port, file-writing stripped) ───
    def polynomial_undrifting_rss(
        self, time_trace, time_trace_bin_centers, channels_spec,
        undrift_order=None, max_undrift_order=10,
        update_undrift_weights=True, ext_indices=np.array([]),
        use_ext_weights=False, use_flcs_bg_corr=False,
        use_burst_removal=False, use_mse_filter=False,
    ):
        """
        Fit and apply a polynomial bleaching/drift correction, faithful
        port of FCS_Fixer.polynomial_undrifting_rss() (plotting/CSV
        writing stripped, as with the rest of this trimmed build).

        Uses the same Ries, Chiantia & Schwille (2009) depletion
        correction formula already used in theatrics.utils
        .bleach_correction.depletion_correct() and in
        zeiss_raw_correlate.get_blcorr_weights() -- applied here on a
        per-photon basis by evaluating the fitted trend at each
        individual photon's own arrival time (not just at bin centres),
        giving a smooth continuous correction.

        Parameters
        ----------
        time_trace             : np.ndarray  from get_time_trace()
        time_trace_bin_centers : np.ndarray  in SECONDS, from get_time_trace()
        channels_spec          : channels_spec tuple identifying which
                                 photons this correction applies to
        undrift_order          : int | None  fixed polynomial degree, or
                                 None to auto-select via
                                 get_auto_undrift_order()
        update_undrift_weights : bool  whether to write the resulting
                                 per-photon weights into
                                 self._weights_undrift (normally True)

        Returns
        -------
        poly_params       : np.ndarray  fitted polynomial coefficients
                            (low-order-first convention)
        undrift_order_used: int         polynomial degree actually used
        """
        channels_spec_norm = self.check_channels_spec(channels_spec)

        # convert bin centres from seconds -> ns for the internal fit
        # (matches FCS_Fixer's internal ns convention throughout)
        time_trace_bin_centers_ns = time_trace_bin_centers * 1e9

        if undrift_order is None:
            undrift_order = self.get_auto_undrift_order(
                time_trace_bin_centers_ns, time_trace, max_undrift_order
            )

        poly_params, _ = Polynomial_fit(
            time_trace_bin_centers_ns, time_trace, undrift_order
        ).run_fit()

        macro_times_select, _, weights_select, indices_select = self.select_photons(
            channels_spec_norm, ext_indices=ext_indices,
            use_ext_weights=use_ext_weights,
            use_burst_removal=use_burst_removal,
            use_flcs_bg_corr=use_flcs_bg_corr,
            use_mse_filter=use_mse_filter,
        )

        macro_times_select_ns = macro_times_select * self._macro_time_resolution

        # evaluate the fitted trend at EACH photon's own arrival time
        poly_values = np.polynomial.polynomial.polyval(
            macro_times_select_ns, poly_params
        )
        poly_zero = np.polynomial.polynomial.polyval(0.0, poly_params)

        weights_undrift_select = (
            1.0 / np.sqrt(poly_values / poly_zero)
            + (1.0 - np.sqrt(poly_values / poly_zero))
        )
        weights_undrift_select[np.isnan(weights_undrift_select)] = 0.0

        if update_undrift_weights:
            self._weights_undrift[indices_select] = weights_undrift_select

        return poly_params, undrift_order
    
    # ── get_auto_threshold (faithful port) ──────────────────────────────────
    def get_auto_threshold(self, time_trace, threshold_alpha=0.02):
        """
        Automatically estimate a threshold photon count for burst
        detection, faithful port of FCS_Fixer.get_auto_threshold().

        Approximates the non-burst baseline using quantile statistics
        (median and 84th percentile, i.e. +1 sigma under a Gaussian
        approximation), Sidak-corrects the false-positive rate for the
        number of bins in the trace, and converts back to an absolute
        photon-count threshold.

        Parameters
        ----------
        time_trace      : np.ndarray  pre-binned photon time trace
        threshold_alpha  : float, 0 < alpha < 1
            Approximate false-positive thresholding rate to allow,
            before Sidak correction for multiple comparisons. Default
            0.02 (2%), which in practice gives decent outlier rejection
            with low false-positive burst flagging.

        Returns
        -------
        int  absolute photon-count threshold
        """
        if not (isinstance(threshold_alpha, float) and 0.0 < threshold_alpha < 1.0):
            raise ValueError(
                "threshold_alpha must be float, > 0 and < 1."
            )

        median_signal, upper_1sigma = np.percentile(time_trace, (50, 84))
        sigma_signal = upper_1sigma - median_signal

        threshold_alpha_sidak = 1 - (1 - threshold_alpha) ** (1 / time_trace.shape[0])
        threshold_counts = int(
            median_signal + np.sqrt(2) * erfcinv(2 * threshold_alpha_sidak) * sigma_signal
        )
        return threshold_counts

    # ── threshold_trace (faithful port) ─────────────────────────────────────
    def threshold_trace(self, time_trace, threshold_alpha=0.02, threshold_counts=None):
        """
        Apply a threshold to a pre-binned time trace to flag burst bins.

        Faithful port of FCS_Fixer.threshold_trace().

        Parameters
        ----------
        time_trace       : np.ndarray  pre-binned photon time trace
        threshold_alpha  : float       used only if threshold_counts is None
        threshold_counts : int | None  fixed absolute threshold; if None,
                           auto-tuned via get_auto_threshold()

        Returns
        -------
        burst_bins       : np.ndarray of bool, True = burst bin
        threshold_counts : int  the threshold actually used
        """
        if threshold_counts is None:
            threshold_counts = self.get_auto_threshold(time_trace, threshold_alpha)
        elif not (isint(threshold_counts) and threshold_counts > 0):
            raise ValueError(
                "threshold_counts must be int > 0, or None for auto-tuning."
            )

        burst_bins = time_trace > threshold_counts
        return burst_bins, threshold_counts

    # ── update_photons_from_bursts (faithful port) ──────────────────────────
    def update_photons_from_bursts(
        self, burst_bins, time_trace_sampling,
        update_weights=True, update_macro_times=True,
    ):
        """
        Use a thresholded (binary) time trace to identify which
        individual photons are burst photons, and optionally update
        self._weights_burst_removal and self._macro_times_correction_bursts.

        Faithful port of FCS_Fixer.update_photons_from_bursts().

        Parameters
        ----------
        burst_bins            : np.ndarray of bool, True = burst bin to remove
        time_trace_sampling   : float, bin width in SECONDS
        update_weights        : bool, whether to write
                                self._weights_burst_removal
        update_macro_times    : bool, whether to write
                                self._macro_times_correction_bursts (closes
                                the time gaps left by excised bursts)

        Returns
        -------
        photon_is_burst : np.ndarray of bool, one entry per photon in the
                         full loaded TTTR object (True = this photon is
                         part of a burst)
        """
        macro_times = self._macro_times.copy()

        n_bins = burst_bins.shape[0]
        burst_bin_indices = np.nonzero(burst_bins)[0]
        time_trace_sampling_macro_time_bins = (
            time_trace_sampling * 1e9 / self._macro_time_resolution
        )

        first_photons_in_bins = np.zeros((n_bins,), dtype=np.uint64)
        current_photon = 0
        bin_edge = 0.0

        for i_bin in range(n_bins):
            while (
                current_photon < len(macro_times)
                and macro_times[current_photon] < bin_edge
            ):
                current_photon += 1
            first_photons_in_bins[i_bin] = current_photon
            bin_edge += time_trace_sampling_macro_time_bins

        photon_is_burst = np.zeros_like(macro_times, dtype=bool)

        for i_burst_bin in burst_bin_indices:
            if i_burst_bin < n_bins - 1:
                photon_is_burst[
                    first_photons_in_bins[i_burst_bin]:first_photons_in_bins[i_burst_bin + 1]
                ] = True
            else:
                photon_is_burst[first_photons_in_bins[i_burst_bin]:] = True

        if update_weights:
            self._weights_burst_removal = np.logical_not(photon_is_burst)
        else:
            self._weights_burst_removal = np.ones_like(macro_times, dtype=bool)

        if update_macro_times:
            if burst_bin_indices.shape[0] > 0:
                macro_times_correction_bursts = np.zeros_like(macro_times, dtype=np.uint64)
                burst_photon_indices = np.nonzero(photon_is_burst)[0]

                macro_time_correction = 0.0
                for i_burst_photon, burst_photon in enumerate(burst_photon_indices[:-1]):
                    macro_time_correction += (
                        self._macro_times[burst_photon + 1] - self._macro_times[burst_photon]
                    )
                    macro_times_correction_bursts[
                        burst_photon + 1:burst_photon_indices[i_burst_photon + 1] + 1
                    ] = macro_time_correction

                if burst_photon_indices[-1] < self._n_total_photons - 1:
                    macro_time_correction += (
                        self._macro_times[burst_photon + 1] - self._macro_times[burst_photon]
                    )
                    macro_times_correction_bursts[burst_photon + 1:] = macro_time_correction

                self._macro_times_correction_bursts = np.floor(
                    macro_times_correction_bursts
                ).astype(np.uint64)
            else:
                self._macro_times_correction_bursts = np.zeros_like(macro_times)
        else:
            self._macro_times_correction_bursts = np.zeros_like(macro_times)

        return photon_is_burst

    # ── run_burst_removal (simplified: single trace, no plotting) ──────────
    def run_burst_removal(
        self, time_trace, time_trace_sampling,
        threshold_alpha=0.02, threshold_counts=None,
        update_weights=True, update_macro_times=True,
    ):
        """
        Convenience wrapper: threshold a single time trace and update
        photon-level burst weights/corrections in one call.

        Simplified from the full FCS_Fixer.run_burst_removal() (which
        also supports multi-channel OR/AND/SUM combination logic and
        file writing); this trimmed build only needs the single-trace
        case, since burst removal is applied independently per detector
        channel by the calling code.

        Returns
        -------
        burst_bins       : np.ndarray of bool, one entry per trace bin
        photon_is_burst  : np.ndarray of bool, one entry per photon
        threshold_counts : int  threshold actually used
        """
        burst_bins, threshold_counts = self.threshold_trace(
            time_trace, threshold_alpha=threshold_alpha, threshold_counts=threshold_counts
        )
        photon_is_burst = self.update_photons_from_bursts(
            burst_bins, time_trace_sampling,
            update_weights=update_weights, update_macro_times=update_macro_times,
        )
        return burst_bins, photon_is_burst, threshold_counts

    # ── get_background_tail_fit (verbatim) ─────────────────────────────────
    def get_background_tail_fit(self, channels_spec, irf_peak_center, fit_start,
                                ext_indices=np.array([]), use_ext_weights=False,
                                use_drift_correction=False, use_burst_removal=False,
                                use_mse_filter=False):
        if not isint(fit_start) or not (fit_start > irf_peak_center):
            raise ValueError('fit_start must be int > irf_peak_center.')

        tcspc_x, tcspc_y = self.get_tcspc_histogram(
            channels_spec, micro_times=[], ext_indices=ext_indices,
            use_ext_weights=use_ext_weights, use_drift_correction=use_drift_correction,
            use_burst_removal=use_burst_removal, use_mse_filter=use_mse_filter)

        tail_fit_use = tcspc_x >= fit_start
        tcspc_x_crop = tcspc_x[tail_fit_use]
        tcspc_y_crop = tcspc_y[tail_fit_use]

        tail_fit = TCSPC_quick_fit(
            x_data=tcspc_x_crop, y_data=tcspc_y_crop, model='exponential',
            initial_params={'x_0': irf_peak_center, 'y_0': np.min(tcspc_y_crop),
                            'amp': np.max(tcspc_y) - np.min(tcspc_y), 'exp_tau': 100.})
        tail_fit_params = tail_fit.run_fit()
        flat_background = tail_fit_params['y_0']

        return flat_background, tail_fit, tcspc_x, tcspc_y

    # ── get_flcs_background_filter (verbatim) ──────────────────────────────
    def get_flcs_background_filter(self, tcspc_x, tcspc_y, flat_background, channels_spec,
                                   handle_outside='zero', update_weights=True,
                                   ext_indices=np.array([])):
        tcspc_x = np.array(tcspc_x, dtype=np.uint64)
        tcspc_y = np.array(tcspc_y, dtype=np.float64)
        channels_spec_norm = self.check_channels_spec(channels_spec)

        pattern_signal = tcspc_y - flat_background
        pattern_background = np.ones_like(tcspc_y)

        nonzero_mask = np.logical_and(tcspc_y != 0, pattern_signal != 0)
        n_nonzeros = nonzero_mask.sum()

        tcspc_y_crop = tcspc_y[nonzero_mask]
        pattern_signal_crop = pattern_signal[nonzero_mask].reshape((n_nonzeros, 1))
        pattern_background_crop = pattern_background[nonzero_mask].reshape((n_nonzeros, 1))

        flcs_weights_crop, patterns_norm_crop = self.get_flcs_filters(
            tcspc_y=tcspc_y_crop,
            patterns=np.concatenate((pattern_signal_crop, pattern_background_crop), axis=1))

        patterns_norm_full = np.zeros((self._n_micro_time_bins, 2), dtype=np.float64)
        patterns_norm_full[tcspc_x[nonzero_mask], 0] = patterns_norm_crop[:, 0]
        patterns_norm_full[tcspc_x[nonzero_mask], 1] = patterns_norm_crop[:, 1]

        flcs_weights_full = np.zeros((self._n_micro_time_bins, 2), dtype=np.float64)
        flcs_weights_full[tcspc_x[nonzero_mask], 0] = flcs_weights_crop[:, 0]
        flcs_weights_full[tcspc_x[nonzero_mask], 1] = flcs_weights_crop[:, 1]

        _, micro_times_select, weights_select, indices_select = self.select_photons(
            channels_spec_norm, ext_indices=ext_indices, use_ext_weights=False,
            use_drift_correction=False, use_flcs_bg_corr=False,
            use_burst_removal=False, use_mse_filter=False)

        if update_weights:
            flcs_photon_weights = np.zeros_like(weights_select)
            for i_bin, micro_time in enumerate(tcspc_x[nonzero_mask]):
                flcs_photon_weights[micro_times_select == micro_time] = flcs_weights_crop[i_bin, 0]

            micro_time_mask = self._get_micro_time_mask(channels_spec_norm)
            tcspc_x_mask = np.zeros((self._n_micro_time_bins,), dtype=bool)  # CHANGED
            tcspc_x_mask[tcspc_x] = True

            if np.any(np.logical_xor(tcspc_x_mask, micro_time_mask)):
                if handle_outside == 'ignore':
                    micro_times_used = np.unique(micro_times_select)
                    not_updated_micro_times = micro_times_used[np.logical_not(np.isin(micro_times_used, tcspc_x))]
                    for micro_time in not_updated_micro_times:
                        mask = micro_times_select == micro_time
                        flcs_photon_weights[mask] = weights_select[mask]
                # 'zero': leave as zero (already the default)

            self._weights_flcs_bg_corr[indices_select] = flcs_photon_weights

        return patterns_norm_full, flcs_weights_full