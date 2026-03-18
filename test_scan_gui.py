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


# ── Calibration table (mirrors scan_gui._load_cal_table) ─────────────────── #

def _load_cal_table(path=None):
    if path is None:
        path = os.path.join(FIXTURE_DIR, "Data.txt")
    freqs, factors = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                freqs.append(float(parts[0]))
                factors.append(float(parts[1]))
    return np.array(freqs), np.array(factors)

_CAL_FREQS, _CAL_FACTORS = _load_cal_table()
RHO_C = 1.54e6   # acoustic impedance of water (Pa·s/m)


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
                    v_rms, frequency, cal_factor, pii, pd
    """
    v_ac = voltage - voltage.mean()

    # Frequency: zero-crossing counter with bidirectional amplitude check.
    # Falls back to peak-to-trough half-period for single-cycle bursts.
    threshold = 0.15 * np.max(np.abs(v_ac)) if np.max(np.abs(v_ac)) > 0 else 0.0
    raw_crosses = np.where((v_ac[:-1] < 0) & (v_ac[1:] >= 0))[0]
    crosses = [i for i in raw_crosses
               if np.max(v_ac[max(0, i - 50):i + 51]) >  threshold
               and np.min(v_ac[max(0, i - 50):i + 51]) < -threshold]

    if len(crosses) >= 2:
        refined = []
        for i in crosses:
            frac = -v_ac[i] / (v_ac[i + 1] - v_ac[i])
            refined.append((i + frac) * dt)
        freq = float(1.0 / np.mean(np.diff(refined)))
    else:
        # Single-cycle fallback: peak-to-trough half-period
        idx_peak   = int(np.argmax(v_ac))
        idx_trough = int(np.argmin(v_ac))
        half_period = abs(idx_peak - idx_trough) * dt
        freq = float(1.0 / (2 * half_period)) if half_period > 0 else 0.0

    cal_idx    = int(np.argmin(np.abs(_CAL_FREQS - freq / 1e6)))
    cal_factor = float(_CAL_FACTORS[cal_idx])
    pii = float(np.sum((v_ac / (cal_factor * 1e-6)) ** 2) * dt) / (RHO_C * 1e4)

    # Pulse duration: 10-90% cumulative energy × 1.25 correction factor
    cumulative   = np.cumsum(v_ac ** 2) * dt
    total_energy = cumulative[-1]
    t1 = int(np.searchsorted(cumulative, 0.10 * total_energy)) * dt
    t2 = int(np.searchsorted(cumulative, 0.90 * total_energy)) * dt
    pd = float((t2 - t1) * 1.25)

    return {
        "n_samples":  len(voltage),
        "timebase_s": dt,
        "v_min":      float(voltage.min()),
        "v_max":      float(voltage.max()),
        "v_pp":       float(voltage.max() - voltage.min()),
        "v_mean":     float(voltage.mean()),
        "v_rms":      float(np.sqrt(np.mean(v_ac ** 2))),
        "frequency":  freq,
        "cal_factor": cal_factor,
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
        assert stats_waveform_1cycle["frequency"] == pytest.approx(3076923.076923077)

    def test_cal_factor(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["cal_factor"] == pytest.approx(0.04754)

    def test_pii(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["pii"] == pytest.approx(0.016296699403268815)

    def test_pd(self, stats_waveform_1cycle):
        assert stats_waveform_1cycle["pd"] == pytest.approx(2.78125e-07)


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
        assert stats_rules["frequency"] == pytest.approx(3000750.187546887)

    def test_cal_factor(self, stats_rules):
        assert stats_rules["cal_factor"] == pytest.approx(0.04583)

    def test_pii(self, stats_rules):
        assert stats_rules["pii"] == pytest.approx(0.052654681896880495)

    def test_pd(self, stats_rules):
        assert stats_rules["pd"] == pytest.approx(1.0200000000000002e-06)


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

    def test_pd_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["pd"] > 0

    def test_pd_less_than_window_duration(self, request, fixture_name):
        """pd must not exceed the total capture window × 1.25."""
        s = request.getfixturevalue(fixture_name)
        window = s["n_samples"] * s["timebase_s"]
        assert s["pd"] <= window * 1.25 + 1e-12

    def test_frequency_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["frequency"] > 0

    def test_v_max_greater_than_v_min(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["v_max"] > s["v_min"]

    def test_cal_factor_positive(self, request, fixture_name):
        s = request.getfixturevalue(fixture_name)
        assert s["cal_factor"] > 0

    def test_pii_units_range(self, request, fixture_name):
        """PII should be in J/cm² range: 1e-5 to 1 for typical ultrasound signals."""
        s = request.getfixturevalue(fixture_name)
        assert 1e-5 < s["pii"] < 1.0


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
        """Zero-crossing frequency detection accurate to <1% for many-cycle sine."""
        dt, n, freq_in = 1e-9, 8192, 5e6
        t = np.arange(n) * dt
        voltage = np.sin(2 * np.pi * freq_in * t)
        s = compute_stats(voltage, dt)
        assert s["frequency"] == pytest.approx(freq_in, rel=0.01)

    def test_single_cycle_burst_frequency(self):
        """Peak-to-trough fallback gives <5% error for a 1-cycle burst."""
        dt, freq_in, amplitude = 1e-9, 2e6, 1.0
        period_samples = int(1 / (freq_in * dt))
        # One full cycle surrounded by zeros
        t_burst = np.arange(period_samples) * dt
        burst = amplitude * np.sin(2 * np.pi * freq_in * t_burst)
        voltage = np.concatenate([np.zeros(500), burst, np.zeros(500)])
        s = compute_stats(voltage, dt)
        assert s["frequency"] == pytest.approx(freq_in, rel=0.05)

    def test_pii_positive_for_ac_signal(self):
        """Any non-zero AC signal produces positive PII."""
        dt, n, freq = 1e-9, 4096, 3e6
        t = np.arange(n) * dt
        voltage = 0.5 * np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        assert s["pii"] > 0

    def test_pii_scales_with_amplitude_squared(self):
        """Doubling amplitude quadruples PII (held frequency → same cal_factor)."""
        dt, n, freq = 1e-9, 8192, 3e6
        t = np.arange(n) * dt
        s1 = compute_stats(1.0 * np.sin(2 * np.pi * freq * t), dt)
        s2 = compute_stats(2.0 * np.sin(2 * np.pi * freq * t), dt)
        assert s2["pii"] == pytest.approx(4.0 * s1["pii"], rel=1e-6)

    def test_pd_positive_for_ac_signal(self):
        """PD must be positive for any AC signal."""
        dt, n, freq = 1e-9, 4096, 3e6
        t = np.arange(n) * dt
        voltage = np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        assert s["pd"] > 0

    def test_pd_bounded_by_window(self):
        """PD × (1/1.25) cannot exceed the capture window duration."""
        dt, n, freq = 1e-9, 4096, 3e6
        t = np.arange(n) * dt
        voltage = np.sin(2 * np.pi * freq * t)
        s = compute_stats(voltage, dt)
        assert s["pd"] / 1.25 <= n * dt + 1e-15

    def test_v_pp_invariant(self):
        """v_pp = v_max − v_min for any waveform."""
        voltage = np.random.default_rng(0).standard_normal(500)
        s = compute_stats(voltage, 1e-9)
        assert s["v_pp"] == pytest.approx(s["v_max"] - s["v_min"], abs=1e-12)

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

    def test_csv_pii_positive(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["pii"]) > 0.0

    def test_csv_pd_positive(self):
        csv_path = os.path.join(FIXTURE_DIR, "scan_results.csv")
        if not os.path.exists(csv_path):
            pytest.skip("scan_results.csv not present")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["pd"]) > 0.0


# ── Scanx result file tests ───────────────────────────────────────────────── #

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
        """Frequency (MHz) should be in 1.5–4.0 MHz range."""
        for row in rows:
            assert 1.5 < row[6] < 4.0

    def test_freq_mhz_consistent_across_x(self, rows):
        """Frequency should vary by less than 5% across X positions."""
        freqs = [row[6] for row in rows]
        avg = sum(freqs) / len(freqs)
        for f in freqs:
            assert abs(f - avg) / avg < 0.05

    def test_pii_positive(self, rows):
        """Column 7 (PII, J/cm²) must be positive."""
        for row in rows:
            assert row[7] > 0.0

    def test_pii_in_expected_range(self, rows):
        """PII (J/cm²) should be in 1e-3 to 1e-1 range for typical signals."""
        for row in rows:
            assert 1e-3 < row[7] < 0.1

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
    with running Waveform-1cycle.txt.txt through compute_stats().

    Column mapping in Scanx-onecycle.txt.txt:
        col3 = v_pp       (V)
        col4 = v_max      (V)
        col5 = v_min      (V)
        col6 = frequency  (MHz)
        col7 = pii        (J/cm²)
        col8 = pd         (s)
    """

    TOLERANCE_VPP  = 0.05   # 5%
    TOLERANCE_FREQ = 0.15   # 15% — Scanx captured at different times/positions
    TOLERANCE_PII  = 0.15   # 15%
    TOLERANCE_PD   = 0.30   # 30% — pd sensitive to burst shape

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

    def test_each_row_vpp_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        ref_vpp = ref_stats["v_pp"]
        for i, row in enumerate(scanx_rows):
            rel_err = abs(row[3] - ref_vpp) / ref_vpp
            assert rel_err < self.TOLERANCE_VPP, (
                f"Row {i}: scanx v_pp={row[3]:.4f} V vs waveform v_pp={ref_vpp:.4f} V "
                f"(rel err={rel_err:.1%})"
            )

    def test_each_row_pii_within_tolerance_of_waveform(self, scanx_rows, ref_stats):
        ref_pii = ref_stats["pii"]
        for i, row in enumerate(scanx_rows):
            rel_err = abs(row[7] - ref_pii) / ref_pii
            assert rel_err < self.TOLERANCE_PII, (
                f"Row {i}: scanx pii={row[7]:.4e} J/cm² vs waveform "
                f"pii={ref_pii:.4e} J/cm² (rel err={rel_err:.1%})"
            )

    def test_vmax_positive_all_rows(self, scanx_rows):
        for i, row in enumerate(scanx_rows):
            assert row[4] > 0.0, f"Row {i}: v_max={row[4]:.4f} V is not positive"

    def test_vmin_negative_all_rows(self, scanx_rows):
        for i, row in enumerate(scanx_rows):
            assert row[5] < 0.0, f"Row {i}: v_min={row[5]:.4f} V is not negative"
