"""Spectral methods for Curie-depth estimation from synthetic magnetic data.
Implementation based on recommendations from: Review of Curie point depth determination through different spectral
methods applied to magnetic data (2020) """

import numpy as np

from forward_simulation import ForwardSimulation


class SpectralMethods:
    """Class to handle spectral methods for estimating Curie depth."""

    PAPER_CENTROID_K_RANGE = (0.0, 0.00025)
    PAPER_TOP_K_RANGE = (0.0003, 0.0014) 

    def __init__(self, forward_simulation: ForwardSimulation):
        self.forward_simulation = forward_simulation

    def _get_grid_shape(self):
        """Return grid shape as (ny, nx) from survey axes."""
        ny = len(self.forward_simulation.y_survey)
        nx = len(self.forward_simulation.x_survey)
        return ny, nx

    def get_gridded_data(self):
        """Return magnetic data reshaped to a 2D survey grid."""
        ny, nx = self._get_grid_shape()
        if self.forward_simulation.dpred.size != ny * nx:
            raise ValueError(
                "dpred size does not match survey grid shape. "
                f"Expected {ny * nx}, got {self.forward_simulation.dpred.size}."
            )
        return self.forward_simulation.dpred.reshape((ny, nx))

    def _hann2d(self, ny, nx):
        """Create a separable 2D Hann window."""
        win_y = np.hanning(ny)
        win_x = np.hanning(nx)
        return win_y[:, np.newaxis] * win_x[np.newaxis, :]

    def preprocess_data(self, data_2d, remove_mean=True, apply_window=True):
        """Preprocess gridded data before FFT."""
        data = data_2d.astype(float).copy()
        if remove_mean:
            data -= np.mean(data)
        if apply_window:
            ny, nx = data.shape
            data *= self._hann2d(ny, nx)
        return data

    def get_frequency_spectrum(self, remove_mean=True, apply_window=True):
        """Compute the shifted 2D FFT spectrum from gridded magnetic data."""
        data_2d = self.get_gridded_data()
        data_processed = self.preprocess_data(
            data_2d, remove_mean=remove_mean, apply_window=apply_window
        )
        spectrum = np.fft.fft2(data_processed)
        return np.fft.fftshift(spectrum)

    def get_power_spectrum(self, remove_mean=True, apply_window=True):
        """Compute shifted 2D power spectrum from magnetic data."""
        spectrum_shifted = self.get_frequency_spectrum(
            remove_mean=remove_mean, apply_window=apply_window
        )
        return np.abs(spectrum_shifted) ** 2

    def get_radial_wavenumber_grid(self):
        """Return radial wavenumber grid in rad/m for the current survey geometry."""
        ny, nx = self._get_grid_shape()

        dx = float(self.forward_simulation.survey_x_spacing)
        dy = float(self.forward_simulation.survey_y_spacing)
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

        kx_shift = np.fft.fftshift(kx)
        ky_shift = np.fft.fftshift(ky)
        kx_grid, ky_grid = np.meshgrid(kx_shift, ky_shift)
        return np.sqrt(kx_grid**2 + ky_grid**2)

    def radial_average_power(self, power_2d, n_bins=None):
        """Compute radial average power as a function of radial wavenumber."""
        kr = self.get_radial_wavenumber_grid()
        kr_flat = kr.ravel()
        power_flat = power_2d.ravel()

        valid = kr_flat > 0.0
        kr_flat = kr_flat[valid]
        power_flat = power_flat[valid]

        if n_bins is None:
            ny, nx = power_2d.shape
            n_bins = max(8, min(nx, ny) // 2)

        k_min = np.min(kr_flat)
        k_max = np.max(kr_flat)
        edges = np.linspace(k_min, k_max, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        radial_power = np.full_like(centers, np.nan, dtype=float)
        for i in range(len(centers)):
            in_bin = (kr_flat >= edges[i]) & (kr_flat < edges[i + 1])
            if np.any(in_bin):
                radial_power[i] = np.mean(power_flat[in_bin])

        keep = np.isfinite(radial_power) & (radial_power > 0.0)
        return centers[keep], radial_power[keep]

    def _fit_log_linear(self, k, y, k_min=None, k_max=None, range_name=None):
        """Fit y = a + b*k over a selected wavenumber range."""
        mask = np.ones_like(k, dtype=bool)
        if k_min is not None:
            mask &= k >= k_min
        if k_max is not None:
            mask &= k <= k_max

        if np.sum(mask) < 3:
            window_name = range_name or "selected"
            window_desc = (
                f"[{k_min}, {k_max}]"
                if (k_min is not None or k_max is not None)
                else "all available k"
            )
            raise ValueError(
                f"Not enough points in {window_name} wavenumber range {window_desc} for linear fit. "
                f"Found {int(np.sum(mask))} point(s); need at least 3."
            )

        coeff = np.polyfit(k[mask], y[mask], 1)
        slope = coeff[0]
        intercept = coeff[1]
        return slope, intercept, mask

    def estimate_top_depth(self, k, radial_power, k_min=None, k_max=None):
        """Estimate top depth Zt using ln(sqrt(P)) = B - Zt*k."""
        y = np.log(np.sqrt(radial_power))
        slope, intercept, mask = self._fit_log_linear(
            k, y, k_min=k_min, k_max=k_max, range_name="top"
        )
        z_top = -slope
        return {
            "z_top": z_top,
            "slope": slope,
            "intercept": intercept,
            "fit_mask": mask,
            "fit_x": k[mask],
            "fit_y": y[mask],
        }

    def estimate_centroid_depth(self, k, radial_power, k_min=None, k_max=None):
        """Estimate centroid depth Zo using ln(sqrt(P)/k) = D - Zo*k."""
        positive_k = k > 0.0
        k = k[positive_k]
        radial_power = radial_power[positive_k]

        y = np.log(np.sqrt(radial_power) / k)
        slope, intercept, mask = self._fit_log_linear(
            k, y, k_min=k_min, k_max=k_max, range_name="centroid"
        )
        z_centroid = -slope
        return {
            "z_centroid": z_centroid,
            "slope": slope,
            "intercept": intercept,
            "fit_mask": mask,
            "fit_x": k[mask],
            "fit_y": y[mask],
        }

    @staticmethod
    def _is_physical_depth_order(z_top, z_centroid, z_bottom):
        """Return True when Zt < Zo < Zb and all depths are positive."""
        return (z_top > 0.0) and (z_centroid > z_top) and (z_bottom > z_centroid)

    def estimate_curie_depth(
        self,
        n_bins=None,
        top_k_range=None,
        centroid_k_range=None,
        remove_mean=True,
        apply_window=True,
        reject_non_physical=True,
        use_paper_ranges=False,
    ):
        """Estimate Zt, Zo, and Zb using Tanaka/Okubo spectral relationships.

        Args:
            n_bins: Number of radial bins for averaging power.
            top_k_range: Tuple (k_min, k_max) in rad/m for top-depth fit.
            centroid_k_range: Tuple (k_min, k_max) in rad/m for centroid fit.
            remove_mean: Remove map mean before FFT.
            apply_window: Apply 2D Hann window before FFT.
            reject_non_physical: Raise an error for non-physical depth ordering.
            use_paper_ranges: If True, use fixed paper-specified wavenumber ranges
                (Z₀: 0–0.05 rad/km, Z_t: 0.1–0.5 rad/km), ignoring top_k_range and
                centroid_k_range parameters.
        """
        ny, nx = self._get_grid_shape()
        base_bins = int(n_bins) if n_bins is not None else max(8, min(nx, ny) // 2)

        if use_paper_ranges:
            # Use paper-specified wavenumber ranges; disable percentile fallback
            trial_settings = [
                {
                    "n_bins": base_bins,
                    "top_k_range": self.PAPER_TOP_K_RANGE,
                    "centroid_k_range": self.PAPER_CENTROID_K_RANGE,
                    "window_percentiles": None,
                }
            ]
        else:
            # Use user-specified ranges only
            trial_settings = [
                {
                    "n_bins": base_bins,
                    "top_k_range": top_k_range,
                    "centroid_k_range": centroid_k_range,
                    "window_percentiles": None,
                }
            ]

        last_error = None
        for idx, trial in enumerate(trial_settings):
            try:
                power = self.get_power_spectrum(
                    remove_mean=remove_mean, apply_window=apply_window
                )
                k, radial_power = self.radial_average_power(
                    power, n_bins=trial["n_bins"]
                )

                if len(k) < 6:
                    raise ValueError(
                        "Not enough radial bins to estimate Curie depth robustly."
                    )

                trial_centroid_range = trial["centroid_k_range"]
                trial_top_range = trial["top_k_range"]

                if trial_centroid_range is None or trial_top_range is None:
                    raise ValueError(
                        "Wavenumber ranges for centroid and top depth must be specified. "
                        "Either provide top_k_range and centroid_k_range, or use use_paper_ranges=True."
                    )

                centroid_mask = (k >= trial_centroid_range[0]) & (
                    k <= trial_centroid_range[1]
                )
                top_mask = (k >= trial_top_range[0]) & (k <= trial_top_range[1])

                print(
                    "Curie-depth fit diagnostics: "
                    f"n_bins={trial['n_bins']}, "
                    f"centroid_range=[{trial_centroid_range[0]:.6g}, {trial_centroid_range[1]:.6g}], "
                    f"centroid_points={int(np.sum(centroid_mask))}, "
                    f"top_range=[{trial_top_range[0]:.6g}, {trial_top_range[1]:.6g}], "
                    f"top_points={int(np.sum(top_mask))}"
                )

                top_fit = self.estimate_top_depth(
                    k, radial_power, k_min=trial_top_range[0], k_max=trial_top_range[1]
                )
                centroid_fit = self.estimate_centroid_depth(
                    k,
                    radial_power,
                    k_min=trial_centroid_range[0],
                    k_max=trial_centroid_range[1],
                )

                z_top = top_fit["z_top"]
                z_centroid = centroid_fit["z_centroid"]
                z_bottom = 2.0 * z_centroid - z_top
                is_physical = self._is_physical_depth_order(z_top, z_centroid, z_bottom)

                print(
                    "Curie-depth fit results:\n "
                    f"z_top={z_top:.3f},\n "
                    f"z_centroid={z_centroid:.3f}, \n"
                    f"z_bottom={z_bottom:.3f}, \n"
                    f"is_physical={is_physical},\n "
                    f"top_fit_points={int(np.sum(top_fit['fit_mask']))}, \n"
                    f"centroid_fit_points={int(np.sum(centroid_fit['fit_mask']))}\n"
                )

                if reject_non_physical and not is_physical:
                    raise ValueError(
                        "Non-physical Curie-depth result from spectral fit: "
                        f"z_top={z_top:.3f}, z_centroid={z_centroid:.3f}, "
                        f"z_bottom={z_bottom:.3f}."
                    )

                return {
                    "k": k,
                    "radial_power": radial_power,
                    "z_top": z_top,
                    "z_centroid": z_centroid,
                    "z_bottom": z_bottom,
                    "top_fit": top_fit,
                    "centroid_fit": centroid_fit,
                    "top_k_range": trial_top_range,
                    "centroid_k_range": trial_centroid_range,
                    "is_physical": is_physical,
                }
            except ValueError as exc:
                last_error = exc
                continue

        raise ValueError(
            "Unable to estimate Curie-depth with specified wavenumber ranges. "
            f"Last error: {last_error}"
        )


if __name__ == "__main__":
    from forward_simulation import ForwardSimulation
    from pathlib import Path

    PROJECT_DIR = Path(__file__).parent

    forward_sim = ForwardSimulation(
        config_yaml=PROJECT_DIR / "spectral_analysis_forward_model.yml",
        randomize_model = True
    )

    spectral_estimator = SpectralMethods(forward_sim)

    power = spectral_estimator.get_power_spectrum(remove_mean=True, apply_window=True)
    k, p = spectral_estimator.radial_average_power(
        power, n_bins=10
    )  # try 16-24 for larger maps

    print("k min/max:", k.min(), k.max(), "n:", len(k))
    print("k bins:", k)

    # Pick non-overlapping branches by index
    i1, i2 = int(0.10 * len(k)), int(0.35 * len(k))  # centroid (low-k)
    j1, j2 = int(0.60 * len(k)), int(0.85 * len(k))  # top (higher-k)

    # Ensure at least 3 points each
    i2 = max(i2, i1 + 2)
    j2 = max(j2, j1 + 2)

    eps = 1e-9
    centroid_k_range = (k[i1] - eps, k[i2] + eps)
    top_k_range = (k[j1] - eps, k[j2] + eps)
    curie_depth_results = spectral_estimator.estimate_curie_depth(
        n_bins=64,
        remove_mean=True,
        apply_window=True,
        use_paper_ranges=True,
    )
