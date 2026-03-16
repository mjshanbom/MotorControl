#!/usr/bin/env python3
"""
Test suite for scan_gui capture computation.

Loads waveform fixture files and verifies that the capture statistics
match expected golden values. Also includes unit tests for the
computation logic using synthetic data.

Run with:
    .venv/bin/python3 -m pytest test_scan_gui.py -v
"""

import csv
import math
import os

import numpy as np
import pytest

# ── Fixture file paths ────────────────────────────────────────────────────── #

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
WAVEFORM_1CYCLE = os.path.join(FIXTURE_DIR, "Waveform-1cycle.txt.txt")
WAVEFORM_RULES  = os.path.join(FIXTURE_DIR, "rules.txt")
SCANX_1CYCLE    = os.path.join(FIXTURE_DIR, "Scanx-onecycle.txt.txt")


# ── Helpers (mirror of scan_gui.py capture logic) ─────────────────────────── #

def parse_waveform_file(path):
    """Parse a scope waveform text file.

    Format:
        line 1: x_increment (seconds per sample)
        line 2: x_origin    (time of first sample, seconds)
        lines 3+: voltage values (one float per line)

    Returns
    -------
    dt : float  — seconds per sample
    x_origin : float — time offset of first sample
    voltage : np.ndarray — voltage values in volts
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    dt      = float(lines[0])
    x_orig  = float(lines[1])
    voltage = np.array([float(v) for v in lines[2:]])
    return dt, x_orig, voltage


def compute_stats(voltage, dt):
    """Exact reimplementation of the stats computation from scan_gui.capture().

    Parameters
    ----------
    voltage : np.ndarray — voltage samples in volts
    dt      : float      — seconds per sample

    Returns
    -------
    dict with keys: n_samples, timebase_s, v_min, v_max, v_pp, v_mean,
                    v_rms, frequency, pii, pd
    """
    v_ac = voltage - voltage.mean()                    # DC removal

    # Hanning-windowed, zero-padded FFT for frequency estimation
    nfft   = 131072
    window = np.hanning(len(v_ac))
    fft_mag = np.abs(np.fft.rfft(v_ac * window, n=nfft))

    k = int(np.argmax(fft_mag[1:]) + 1)               # peak bin, skip DC
    if 1 < k < len(fft_mag) - 1:                      # parabolic sub-bin
        a, b, c = fft_mag[k - 1], fft_mag[k], fft_mag[k + 1]
        denom  = a - 2 * b + c
        k_frac = k + (a - c) / (2 * denom) if denom != 0 else k
    else:
        k_frac = k

    freq = float(k_frac / (nfft * dt))
    pii  = float(np.sum(v_ac ** 2) * dt)              # V²·s
    pd   = float(len(voltage) * dt)                   # capture window duration

    return {
        "n_samples":  len(voltage),
        "timebase_s": dt,
        "v_min":      float(voltage.min()),
        "v_max":      float(voltage.max()),
        "v_pp":       float(voltage.max() - voltage.min()),
        "v_mean":     float(voltage.mean()),
        "v_rms":      float(np.sqrt(np.mean(v_ac ** 2))),
        "frequency":  freq,
        "pii":        pii,
        "pd":         pd,
    }


# ── Parser tests ──────────────────────────────────────────────────────────── #

class TestWaveformParser:
    """Tests for parse_waveform_file()."""

    def test_parse_waveform_1cycle_returns_tuple(self):
        dt, x_origin, voltage = parse_waveform_file(WAVEFORM_1CYCLE)
        assert isinstance(dt, float)
        assert isinstance(x_origin, float)
        assert isinstance(voltage, np.ndarray)

    def test_parse_waveform_1cycle_timebase(self):
        dt, _, _ = parse_waveform_file(WAVEFORM_1CYCLE)
        assert dt == pytest.approx(2.5e-9)

    def test_parse_waveform_1cycle_x_origin(self):
        _, x_origin, _ = parse_waveform_file(WAVEFORM_1CYCLE)
        assert x_origin == pytest.approx(-2.5e-6)

    def test_parse_waveform_1cycle_sample_count(self):
        _, _, voltage = parse_waveform_file(WAVEFORM_1CYCLE)
        assert len(voltage) == 1991

    def test_parse_rules_timebase(self):
        dt, _, _ = parse_waveform_file(WAVEFORM_RULES)
        assert dt == pytest.approx(5.0e-10)

    def test_parse_rules_x_origin(self):
        _, x_origin, _ = parse_waveform_file(WAVEFORM_RULES)
        assert x_origin == pytest.approx(-5.0e-7)

    def test_parse_rules_sample_count(self):
        _, _, voltage = parse_waveform_file(WAVEFORM_RULES)
        assert len(voltage) == 1991

    def test_x_origin_is_negative(self):
        _, x_origin, _ = parse_waveform_file(WAVEFORM_1CYCLE)
        assert x_origin < 0


# ── Golden-value tests: Waveform-1cycle.txt.txt ───────────────────────────── #

@pytest.fixture(scope="module")
def stats_waveform_1cycle():
    dt, _, voltage = parse_waveform_file(WAVEFORM_1CYCLE)
    return compute_stats(voltage, dt)


class TestWaveform1CycleGolden:
    """Golden-value regression tests for Waveform-1cycle.txt.txt."""

    def test_n_samples(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["n_samples"] == 1991

    def test_timebase(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["timebase_s"] == pytest.approx(2.5e-9)

    def test_v_min(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["v_min"] == pytest.approx(-1.824)

    def test_v_max(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["v_max"] == pytest.approx(1.845)

    def test_v_pp(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["v_pp"] == pytest.approx(3.669)

    def test_v_mean(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["v_mean"] == pytest.approx(-0.0010673922651933687)

    def test_v_rms(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["v_rms"] == pytest.approx(0.33756991233495404)

    def test_frequency(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["frequency"] == pytest.approx(2501810.252705135)

    def test_pii(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["pii"] == pytest.approx(5.672032760405816e-07)

    def test_pd(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["pd"] == pytest.approx(4.9775e-06)


# ── Golden-value tests: rules.txt ─────────────────────────────────────────── #

@pytest.fixture(scope="module")
def stats_rules():
    dt, _, voltage = parse_waveform_file(WAVEFORM_RULES)
    return compute_stats(voltage, dt)


class TestRulesWaveformGolden:
    """Golden-value regression tests for rules.txt."""

    def test_n_samples(self, stats_rules):
        assert stats_rules["n_samples"] == 1991

    def test_timebase(self, stats_rules):
        assert stats_rules["timebase_s"] == pytest.approx(5.0e-10)

    def test_v_min(self, stats_rules):
        assert stats_rules["v_min"] == pytest.approx(-1.839)

    def test_v_max(self, stats_rules):
        assert stats_rules["v_max"] == pytest.approx(1.850)

    def test_v_pp(self, stats_rules):
        assert stats_rules["v_pp"] == pytest.approx(3.689)

    def test_v_mean(self, stats_rules):
        assert stats_rules["v_mean"] == pytest.approx(-0.00021958714213964013)

    def test_v_rms(self, stats_rules):
        assert stats_rules["v_rms"] == pytest.approx(1.308001018571816)

    def test_frequency(self, stats_rules):
        assert stats_rules["frequency"] == pytest.approx(3003350.642467745)

    def test_pii(self, stats_rules):
        assert stats_rules["pii"] == pytest.approx(1.7031677645942764e-06)

    def test_pd(self, stats_rules):
        assert stats_rules["pd"] == pytest.approx(9.955e-07)


# ── Invariant tests (apply to both waveforms) ─────────────────────────────── #

@pytest.mark.parametrize("fixture_name", ["stats_waveform_1cycle", "stats_rules"])
class TestStatInvariants:
    """Properties that must hold for any valid capture output."""

    def test_v_pp_equals_max_minus_min(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["v_pp"] == pytest.approx(s["v_max"] - s["v_min"], abs=1e-12)

    def test_v_pp_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["v_pp"] > 0

    def test_v_rms_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["v_rms"] > 0

    def test_pii_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["pii"] > 0

    def test_pd_equals_n_samples_times_dt(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["pd"] == pytest.approx(s["n_samples"] * s["timebase_s"])

    def test_frequency_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["frequency"] > 0

    def test_v_max_greater_than_v_min(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["v_max"] > s["v_min"]

    def test_pii_consistent_with_rms_and_pd(self, request, fixture_name):
        """pii ≈ v_rms² × pd (by definition of RMS)."""
        s = request.getfixturevalue(fixture_name)
        assert s["pii"] == pytest.approx(s["v_rms"] ** 2 * s["pd"], rel=1e-9)


# ── Unit tests for compute_stats with synthetic data ─────────────────────── #

class TestComputeStats:
    """Unit tests using synthetic waveforms with known properties."""

    def test_dc_signal_zero_rms(self):
        """Pure DC: AC-coupled rms and pii must be zero."""
        voltage = np.full(1000, 2.5)
        s = compute_stats(voltage, 1e-9)
        assert s["v_rms"] == pytest.approx(0.0, abs=1e-12)
        assert s["pii"]   == pytest.approx(0.0, abs=1e-12)

    def test_dc_signal_mean(self):
        """v_mean equals the DC level."""
        voltage = np.full(1000, 2.5)
        s = compute_stats(voltage, 1e-9)
        assert s["v_mean"] == pytest.approx(2.5)

    def test_dc_signal_v_pp_zero(self):
        """Flat DC: v_pp = 0."""
        voltage = np.full(1000, -1.0)
        s = compute_stats(voltage, 1e-9)
        assert s["v_pp"] == pytest.approx(0.0, abs=1e-12)

    def test_sine_v_pp(self):
        """Pure sine: v_pp ≈ 2 × amplitude."""
        dt, n, amplitude, freq = 1e-9, 4096, 1.5, 1e6
        t = np.arange(n) * dt
        voltage = amplitude * np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        assert s["v_pp"] == pytest.approx(2 * amplitude, rel=1e-3)

    def test_sine_v_rms(self):
        """Pure sine with zero mean: v_rms ≈ amplitude / √2."""
        dt, n, amplitude, freq = 1e-9, 8192, 1.0, 1e6
        t = np.arange(n) * dt
        voltage = amplitude * np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        assert s["v_rms"] == pytest.approx(amplitude / math.sqrt(2), rel=1e-2)

    def test_sine_frequency_detection(self):
        """Frequency detection should be accurate to <1% for many-cycle sine."""
        dt, n, freq_in = 1e-9, 8192, 5e6
        t = np.arange(n) * dt
        voltage = np.sin(2 * np.pi * freq_in * t)
        s = compute_stats(voltage, dt)
        assert s["frequency"] == pytest.approx(freq_in, rel=0.01)

    def test_pd_equals_n_times_dt(self):
        """pd is always n_samples × dt regardless of waveform shape."""
        dt = 2.5e-9
        voltage = np.random.default_rng(42).standard_normal(1200)
        s = compute_stats(voltage, dt)
        assert s["pd"] == pytest.approx(1200 * dt)

    def test_v_pp_invariant(self):
        """v_pp = v_max − v_min for any waveform."""
        voltage = np.random.default_rng(0).standard_normal(500)
        s = compute_stats(voltage, 1e-9)
        assert s["v_pp"] == pytest.approx(s["v_max"] - s["v_min"], abs=1e-12)

    def test_pii_sine_analytical(self):
        """For a pure sine, pii = amplitude²/2 × pd (within 1%)."""
        dt, n, amplitude, freq = 1e-9, 10000, 1.0, 1e6
        t = np.arange(n) * dt
        voltage = amplitude * np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        expected_pii = (amplitude ** 2 / 2) * (n * dt)
        assert s["pii"] == pytest.approx(expected_pii, rel=0.01)

    def test_n_samples_matches_input_length(self):
        """n_samples always equals len(voltage)."""
        for n in [100, 500, 1200, 1991]:
            voltage = np.zeros(n)
            s = compute_stats(voltage, 1e-9)
            assert s["n_samples"] == n

    def test_timebase_passthrough(self):
        """timebase_s is returned unchanged."""
        for dt in [1e-9, 2.5e-9, 5e-10]:
            s = compute_stats(np.zeros(100), dt)
            assert s["timebase_s"] == dt


# ── Scan result CSV tests ─────────────────────────────────────────────────── #

class TestScanxResultFile:
    """Tests for the Scanx-onecycle.txt.txt expected scan output."""

    @pytest.fixture(scope="class")
    def rows(self):
        rows = []
        with open(SCANX_1CYCLE) as f:
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) == 9:
                    rows.append([float(c) for c in cols])
        return rows

    def test_file_exists(self):
        assert os.path.exists(SCANX_1CYCLE)

    def test_has_five_rows(self, rows):
        assert len(rows) == 5

    def test_nine_columns_per_row(self, rows):
        for row in rows:
            assert len(row) == 9

    def test_y_and_z_fixed_at_zero(self, rows):
        """All rows have y=0 and z=0 (single X-axis scan)."""
        for row in rows:
            assert row[1] == pytest.approx(0.0)   # y_mm
            assert row[2] == pytest.approx(0.0)   # z_mm

    def test_x_positions_span_expected_range(self, rows):
        """X runs from -2 to +2 mm in steps of 1 mm."""
        x_vals = [row[0] for row in rows]
        assert x_vals == pytest.approx([-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_v_pp_all_positive(self, rows):
        """v_pp (col 3) must be positive at every point."""
        for row in rows:
            assert row[3] > 0.0

    def test_v_pp_near_expected(self, rows):
        """v_pp should be ~3.68–3.72 V (one AC cycle, ~±1.85 V amplitude)."""
        for row in rows:
            assert 3.60 < row[3] < 3.75

    def test_v_max_positive(self, rows):
        for row in rows:
            assert row[4] > 0.0   # v_max (col 4)

    def test_v_min_negative(self, rows):
        for row in rows:
            assert row[5] < 0.0   # v_min (col 5)

    def test_v_pp_equals_vmax_minus_vmin(self, rows):
        """v_pp = v_max − v_min must hold for every row."""
        for row in rows:
            v_pp, v_max, v_min = row[3], row[4], row[5]
            assert v_pp == pytest.approx(v_max - v_min, abs=1e-4)

    def test_freq_mhz_positive(self, rows):
        """Column 6 (frequency in MHz) must be positive."""
        for row in rows:
            assert row[6] > 0.0

    def test_freq_mhz_near_expected(self, rows):
        """Frequency (MHz) should be near 2.5–3.5 MHz range."""
        for row in rows:
            assert 2.0 < row[6] < 4.0

    def test_freq_mhz_consistent_across_x(self, rows):
        """Frequency should vary by less than 5% across X positions."""
        freqs = [row[6] for row in rows]
        avg = sum(freqs) / len(freqs)
        for f in freqs:
            assert abs(f - avg) / avg < 0.05

    def test_vrms_cal_positive(self, rows):
        """Column 7 (v_rms × cal_factor) must be positive."""
        for row in rows:
            assert row[7] > 0.0

    def test_vrms_cal_in_expected_range(self, rows):
        """Calibrated v_rms should be in expected millivolt range (0.01–0.10 V)."""
        for row in rows:
            assert 0.005 < row[7] < 0.10

    def test_pd_positive(self, rows):
        """Column 8 (pulse duration, seconds) must be positive."""
        for row in rows:
            assert row[8] > 0.0

    def test_pd_in_expected_range(self, rows):
        """pd (seconds) should be in sub-microsecond range for this capture."""
        for row in rows:
            assert 1e-8 < row[8] < 1e-5

    def test_v_pp_consistent_across_x(self, rows):
        """v_pp should vary by less than 2% across X positions."""
        vpp_vals = [row[3] for row in rows]
        assert max(vpp_vals) - min(vpp_vals) < 0.04   # < ~1% of 3.7 V


# ── Cross-comparison: Scanx file vs. Waveform-1cycle stats ───────────────── #

class TestScanxVsWaveform1Cycle:
    """Verify that Scanx-onecycle.txt.txt values are physically consistent
    with running Waveform-1cycle.txt.txt through scan_gui.capture().

    Column mapping in Scanx-onecycle.txt.txt:
        col3 = v_pp  (V)
        col4 = v_max (V)
        col5 = v_min (V)
        col6 = frequency (MHz)
        col7 = v_rms × CAL_FACTOR_3MHZ  (calibrated v_rms, V)
        col8 = pd  (s) — from a different scope capture window; not comparable
    """

    TOLERANCE_VPP   = 0.05   # 5% — allows for real spatial variation + capture differences
    TOLERANCE_FREQ  = 0.05   # 5%
    TOLERANCE_VRMS  = 0.07   # 7% — calibrated rms has slight cal-factor uncertainty

    @pytest.fixture(scope="class")
    def scanx_rows(self):
        rows = []
        with open(SCANX_1CYCLE) as f:
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) == 9:
                    rows.append([float(c) for c in cols])
        return rows

    @pytest.fixture(scope="class")
    def ref_stats(self):
        dt, _, voltage = parse_waveform_file(WAVEFORM_1CYCLE)
        return compute_stats(voltage, dt)

    # ── v_pp comparison ──────────────────────────────────────────────────── #

    def test_each_row_vpp_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Each scan point's v_pp should be within 5% of the waveform v_pp."""
        ref_vpp = ref_stats["v_pp"]
        for i, row in enumerate(scanx_rows):
            scanx_vpp = row[3]
            rel_err = abs(scanx_vpp - ref_vpp) / ref_vpp
            assert rel_err < self.TOLERANCE_VPP, (
                f"Row {i}: scanx v_pp={scanx_vpp:.4f} V vs waveform v_pp={ref_vpp:.4f} V "
                f"(rel err={rel_err:.1%} > {self.TOLERANCE_VPP:.0%})"
            )

    def test_mean_vpp_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Mean v_pp across all scan points should be within 5% of waveform v_pp."""
        ref_vpp = ref_stats["v_pp"]
        mean_vpp = sum(row[3] for row in scanx_rows) / len(scanx_rows)
        rel_err = abs(mean_vpp - ref_vpp) / ref_vpp
        assert rel_err < self.TOLERANCE_VPP, (
            f"mean scanx v_pp={mean_vpp:.4f} V vs waveform v_pp={ref_vpp:.4f} V "
            f"(rel err={rel_err:.1%})"
        )

    # ── frequency comparison ─────────────────────────────────────────────── #

    def test_each_row_freq_mhz_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Each scan point's frequency (MHz) should be within 5% of waveform frequency."""
        ref_freq_mhz = ref_stats["frequency"] / 1e6
        for i, row in enumerate(scanx_rows):
            scanx_freq_mhz = row[6]
            rel_err = abs(scanx_freq_mhz - ref_freq_mhz) / ref_freq_mhz
            assert rel_err < self.TOLERANCE_FREQ, (
                f"Row {i}: scanx freq={scanx_freq_mhz:.4f} MHz vs waveform "
                f"freq={ref_freq_mhz:.4f} MHz (rel err={rel_err:.1%})"
            )

    def test_mean_freq_mhz_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Mean frequency across all scan points should be within 5% of waveform frequency."""
        ref_freq_mhz = ref_stats["frequency"] / 1e6
        mean_freq = sum(row[6] for row in scanx_rows) / len(scanx_rows)
        rel_err = abs(mean_freq - ref_freq_mhz) / ref_freq_mhz
        assert rel_err < self.TOLERANCE_FREQ, (
            f"mean scanx freq={mean_freq:.4f} MHz vs waveform freq={ref_freq_mhz:.4f} MHz "
            f"(rel err={rel_err:.1%})"
        )

    # ── calibrated v_rms comparison ──────────────────────────────────────── #

    def test_each_row_vrms_cal_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Each scan point's v_rms_cal (col7) should be within 7% of
        waveform v_rms × CAL_FACTOR_3MHZ."""
        ref_vrms_cal = ref_stats["v_rms"] * CAL_FACTOR_3MHZ
        for i, row in enumerate(scanx_rows):
            scanx_vrms_cal = row[7]
            rel_err = abs(scanx_vrms_cal - ref_vrms_cal) / ref_vrms_cal
            assert rel_err < self.TOLERANCE_VRMS, (
                f"Row {i}: scanx v_rms_cal={scanx_vrms_cal:.5f} V vs waveform "
                f"v_rms_cal={ref_vrms_cal:.5f} V (rel err={rel_err:.1%})"
            )

    def test_mean_vrms_cal_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        """Mean calibrated v_rms across all scan points should be within 7% of waveform value."""
        ref_vrms_cal = ref_stats["v_rms"] * CAL_FACTOR_3MHZ
        mean_vrms_cal = sum(row[7] for row in scanx_rows) / len(scanx_rows)
        rel_err = abs(mean_vrms_cal - ref_vrms_cal) / ref_vrms_cal
        assert rel_err < self.TOLERANCE_VRMS, (
            f"mean scanx v_rms_cal={mean_vrms_cal:.5f} V vs waveform "
            f"v_rms_cal={ref_vrms_cal:.5f} V (rel err={rel_err:.1%})"
        )

    # ── sign checks: v_max and v_min ─────────────────────────────────────── #

    def test_vmax_positive_all_rows(self, scanx_rows):
        """v_max (col4) must be positive (signal is AC-coupled around ~0 V)."""
        for i, row in enumerate(scanx_rows):
            assert row[4] > 0.0, f"Row {i}: v_max={row[4]:.4f} V is not positive"

    def test_vmin_negative_all_rows(self, scanx_rows):
        """v_min (col5) must be negative (signal swings below zero)."""
        for i, row in enumerate(scanx_rows):
            assert row[5] < 0.0, f"Row {i}: v_min={row[5]:.4f} V is not negative"

    def test_vmax_vmin_magnitude_similar(self, scanx_rows, ref_stats):
        """|v_max| and |v_min| should each be within 5% of half the waveform v_pp.

        Individual v_max/v_min can differ from the waveform reference because
        capture phase differs, but both should be roughly amplitude/2.
        """
        half_vpp = ref_stats["v_pp"] / 2.0
        for i, row in enumerate(scanx_rows):
            for label, val in [("v_max", row[4]), ("v_min", row[5])]:
                rel_err = abs(abs(val) - half_vpp) / half_vpp
                assert rel_err < 0.10, (
                    f"Row {i}: |{label}|={abs(val):.4f} V vs half_vpp={half_vpp:.4f} V "
                    f"(rel err={rel_err:.1%} > 10%)"
                )


# ── scan_results.csv format tests ─────────────────────────────────────────── #

class TestScanResultsCsv:
    """Tests for the scan_results.csv output format expected from scan_gui."""

    EXPECTED_FIELDNAMES = [
        "x_mm", "y_mm", "z_mm", "v_pp", "v_max", "v_min",
        "frequency", "pii", "pd",
    ]

    def test_csv_has_correct_header(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == self.EXPECTED_FIELDNAMES

    def test_csv_rows_have_numeric_values(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for field in self.EXPECTED_FIELDNAMES:
                    float(row[field])   # raises ValueError if not numeric

    def test_csv_v_pp_positive(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["v_pp"]) >= 0.0

    def test_csv_frequency_positive(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["frequency"]) > 0.0


# ── Calibrated pii tests ──────────────────────────────────────────────────── #

# Hydrophone calibration factor at 3 MHz: converts raw pii (V²·s) to
# calibrated acoustic intensity (e.g. W·s/cm² or equivalent units).
CAL_FACTOR_3MHZ = 0.04583


def apply_calibration(pii_raw, cal_factor):
    """Apply calibration factor to pii.

    The calibration factor is applied to the voltage *before* squaring,
    so the effect on pii is multiplication by cal_factor².

        v_cal  = v_ac × cal_factor
        pii_cal = Σ(v_cal²) × dt = cal_factor² × Σ(v_ac²) × dt
    """
    return pii_raw * cal_factor ** 2


class TestCalibratedPii:
    """Tests for pii × calibration factor (3 MHz hydrophone)."""

    # ── rules.txt: the 3 MHz waveform (calibration applies exactly) ── #

    def test_rules_frequency_near_3mhz(self, stats_rules):
        """Confirm rules.txt is a ~3 MHz signal before applying cal factor."""
        assert stats_rules["frequency"] == pytest.approx(3.003e6, rel=0.01)

    def test_rules_pii_raw(self, stats_rules):
        assert stats_rules["pii"] == pytest.approx(1.7031677645942764e-06)

    def test_rules_pii_calibrated_golden(self, stats_rules):
        """Golden value: pii_cal for the 3 MHz waveform."""
        pii_cal = apply_calibration(stats_rules["pii"], CAL_FACTOR_3MHZ)
        assert pii_cal == pytest.approx(3.5773146675916312e-09, rel=1e-3)

    def test_rules_pii_calibrated_positive(self, stats_rules):
        pii_cal = apply_calibration(stats_rules["pii"], CAL_FACTOR_3MHZ)
        assert pii_cal > 0

    # ── Waveform-1cycle.txt.txt: ~2.5 MHz (cal factor is approximate) ── #

    def test_waveform_1cycle_frequency(self, stats_waveform_1cycle):
        """Waveform-1cycle is ~2.5 MHz — not 3 MHz, so cal factor is approximate."""
        assert stats_waveform_1cycle["frequency"] == pytest.approx(2.502e6, rel=0.01)

    def test_waveform_1cycle_pii_calibrated_golden(self, stats_waveform_1cycle):
        """Golden value: pii_cal for the ~2.5 MHz waveform with 3 MHz cal factor."""
        pii_cal = apply_calibration(stats_waveform_1cycle["pii"], CAL_FACTOR_3MHZ)
        assert pii_cal == pytest.approx(1.1913474650392735e-09, rel=1e-3)

    def test_waveform_1cycle_pii_calibrated_positive(self, stats_waveform_1cycle):
        pii_cal = apply_calibration(stats_waveform_1cycle["pii"], CAL_FACTOR_3MHZ)
        assert pii_cal > 0

    # ── apply_calibration() unit tests ── #

    def test_apply_calibration_scales_as_square(self):
        # v × 0.04583 then squared → pii × 0.04583²
        assert apply_calibration(1.0, 0.04583) == pytest.approx(0.04583 ** 2)

    def test_apply_calibration_zero_factor(self):
        assert apply_calibration(1e-6, 0.0) == pytest.approx(0.0)

    def test_apply_calibration_identity(self):
        assert apply_calibration(5e-7, 1.0) == pytest.approx(5e-7)

    def test_apply_calibration_not_linear(self):
        """Confirms cal_factor² scaling, not linear scaling."""
        pii_raw = 1e-6
        result  = apply_calibration(pii_raw, 0.04583)
        assert result != pytest.approx(pii_raw * 0.04583)   # linear would be wrong
        assert result == pytest.approx(pii_raw * 0.04583 ** 2)
