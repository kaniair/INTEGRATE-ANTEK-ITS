"""
INTEGRATE - Performance Curve Engine
Rule-based engine for pump & compressor performance curve monitoring.
Standards: ASME PTC 8.2, API 610 (pump), ASME PTC 10, API 617 (compressor).
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from enum import Enum


class PumpZone(Enum):
    POR  = "POR"
    AOR  = "AOR"
    CAVITATION_RISK = "Cavitation"
    MINIMUM_FLOW    = "Min Flow"
    OVERLOAD        = "Overload"


class CompressorZone(Enum):
    SURGE          = "Z1 - Surge Zone"
    PROTECTION     = "Z2 - Protection Zone"
    OPTIMAL        = "Z3 - Zona Operasi Optimal"
    NORMAL         = "Z4 - Zona Operasi Normal"
    LOW_SPEED      = "Z5 - Zona Kecepatan Rendah"
    MIN_SPEED      = "Z6 - Zona Kecepatan Minimum"
    BELOW_ENVELOPE = "Z7 - Di Bawah Envelope"
    STONEWALL      = "Z8 - Stonewall / Choke"


@dataclass
class PumpStatus:
    equipment_id: str
    flow: float
    head: float
    power: float
    npsha: float
    zone: PumpZone
    efficiency: float
    alert_messages: list


@dataclass
class CompressorStatus:
    equipment_id: str
    inlet_flow: float
    pressure_ratio: float
    polytropic_head: float
    shaft_power: float
    poly_efficiency: float
    speed_rpm: float
    zone: CompressorZone
    surge_margin_pct: float
    alert_messages: list


# -------------------------------------------------
# CURVE LOADING
# -------------------------------------------------

def load_pump_curves(curves_path: str = "curves/pump_curves.json") -> dict:
    """
    Load pump curves. Prioritas:
    1. Jika file JSON ada dan sudah format baru (punya 'head_flow' + 'coeff_max_diameter') -> pakai langsung
    2. Jika file JSON ada tapi format lama -> normalize
    3. Jika file tidak ada -> pakai _default_pump_curves()
    """
    if not os.path.exists(curves_path):
        return _default_pump_curves()

    with open(curves_path) as f:
        raw = json.load(f)

    # Cek apakah format sudah baru (sudah punya semua 3 koefisien diameter)
    hf = raw.get("head_flow", {})
    has_all_coeff = (
        "coeff_max_diameter"   in hf and
        "coeff_rated_diameter" in hf and
        "coeff_min_diameter"   in hf and
        "coefficients"         in hf
    )

    if "head_flow" in raw and has_all_coeff:
        # Format baru — langsung return tanpa normalize
        return raw
    elif "head_flow" in raw:
        # Format setengah baru — normalize tapi preserve coeff yang ada
        return _normalize_pump_curves(raw)
    else:
        # Format lama — normalize
        return _normalize_pump_curves(raw)


def _normalize_pump_curves(raw: dict) -> dict:
    """
    Normalize berbagai format pump_curves.json ke format standar.
    Selalu preserve semua 3 koefisien diameter impeller.
    """
    # Support dua format key: 'head_flow' (format baru) atau 'head' (format lama)
    hf = raw.get("head_flow", raw.get("head", {}))
    rated_flow = hf.get("rated_flow", 600.2)

    # ── Batas POR/AOR ────────────────────────────────────────────────
    # Deteksi otomatis apakah nilai dalam % atau GPM absolut
    por_min_raw = hf.get("por_min_flow", 420.0)
    por_max_raw = hf.get("por_max_flow", 711.6)
    aor_min_raw = hf.get("aor_min_flow", 115.0)
    aor_max_raw = hf.get("aor_max_flow", 750.0)

    # Jika nilai > 200 kemungkinan GPM absolut, konversi ke %
    def to_pct(v):
        return v if v <= 200 else v / rated_flow * 100

    por_min = to_pct(por_min_raw)
    por_max = to_pct(por_max_raw)
    aor_min = to_pct(aor_min_raw)
    aor_max = to_pct(aor_max_raw)

    # ── Koefisien kurva head ──────────────────────────────────────────
    # Ambil dari berbagai kemungkinan key, jangan pernah default semua ke rated
    coeff_rated = (
        hf.get("coeff_rated_diameter") or
        hf.get("coefficients") or
        hf.get("rated", {}).get("coefficients") or
        [5e-13, 1e-9, -4e-6, 0.0013, -0.2108, 2146.9]   # fallback Excel data
    )
    coeff_max = (
        hf.get("coeff_max_diameter") or
        hf.get("max_diameter", {}).get("coefficients") or
        [4e-12, -6e-9, 3e-7, 0.0003, -0.1651, 2353.8]    # fallback Excel data
    )
    coeff_min = (
        hf.get("coeff_min_diameter") or
        hf.get("min_diameter", {}).get("coefficients") or
        [-8e-13, 4e-9, -6e-6, 0.0016, -0.1897, 1858.3]   # fallback Excel data
    )

    # ── Power & NPSH ─────────────────────────────────────────────────
    p    = raw.get("power", {})
    npsh = raw.get("npsh", {})

    rated_power = (
        p.get("rated_power") or
        p.get("rated_power_hp") or
        500.0
    )
    power_coeff = (
        p.get("coefficients") or
        [1e-12, -2e-9, 1e-6, 0.0002, 0.1081, 292.41]     # fallback Excel data
    )
    npshr_coeff = (
        npsh.get("npshr_coefficients") or
        [-2e-13, 3e-10, -2e-7, 0.0001, -0.0032, 7.8808]  # fallback Excel data
    )

    # ── Fluid properties ─────────────────────────────────────────────
    fluid = raw.get("fluid", {})
    sg    = (fluid.get("specific_gravity") or
             npsh.get("specific_gravity") or 1.084)
    pv    = (fluid.get("vapor_pressure_psia") or
             npsh.get("vapor_pressure_psia") or 1.02)
    eta   = fluid.get("pump_efficiency", 0.686)

    return {
        "equipment_id": raw.get("equipment_id", "340-P-1005 A/B"),
        "fluid": {
            "specific_gravity"   : sg,
            "vapor_pressure_psia": pv,
            "pump_efficiency"    : eta,
        },
        "head_flow": {
            "rated_flow"          : rated_flow,
            "rated_head"          : hf.get("rated_head", hf.get("rated_head_ft", 1795.9)),
            "por_min_flow"        : por_min,
            "por_max_flow"        : por_max,
            "aor_min_flow"        : aor_min,
            "aor_max_flow"        : aor_max,
            # Semua 3 koefisien diameter — JANGAN pernah default semua ke rated
            "coefficients"        : coeff_rated,   # alias untuk check_pump_performance
            "coeff_rated_diameter": coeff_rated,
            "coeff_max_diameter"  : coeff_max,
            "coeff_min_diameter"  : coeff_min,
        },
        "npsh": {
            "npsha"              : npsh.get("npsha", None),
            "vapor_pressure_psia": pv,
            "specific_gravity"   : sg,
            "npshr_coefficients" : npshr_coeff,
        },
        "power": {
            "rated_power" : rated_power,
            "coefficients": power_coeff,
        },
    }


def _normalize_pump_curves(raw: dict) -> dict:
    """
    Normalize pump_curves.json ke format yang dipakai check_pump_performance().
    Mempertahankan semua koefisien 3 diameter impeller.
    """
    h = raw.get("head_flow", raw.get("head", {}))
    rated_flow = h.get("rated_flow", 600.2)

    # Batas POR/AOR — sudah dalam % jika dari _default_pump_curves()
    # atau dalam GPM absolut jika dari format lama, deteksi otomatis
    por_min_raw = h.get("por_min_flow", 420.0)
    por_max_raw = h.get("por_max_flow", 711.6)
    aor_min_raw = h.get("aor_min_flow", 115.0)
    aor_max_raw = h.get("aor_max_flow", 750.0)

    # Jika nilai > 200, kemungkinan GPM absolut bukan persen → konversi
    por_min = por_min_raw if por_min_raw <= 200 else por_min_raw / rated_flow * 100
    por_max = por_max_raw if por_max_raw <= 200 else por_max_raw / rated_flow * 100
    aor_min = aor_min_raw if aor_min_raw <= 200 else aor_min_raw / rated_flow * 100
    aor_max = aor_max_raw if aor_max_raw <= 200 else aor_max_raw / rated_flow * 100

    # Koefisien — ambil semua diameter jika ada
    coeff_rated = (h.get("coefficients") or
                   h.get("coeff_rated_diameter") or
                   [-0.0012, -0.1, 1954.0])
    coeff_max   = h.get("coeff_max_diameter",   coeff_rated)
    coeff_min   = h.get("coeff_min_diameter",   coeff_rated)

    p = raw.get("power", {})
    rated_power = p.get("rated_power", 500.0)

    npsh = raw.get("npsh", {})

    return {
        "equipment_id": raw.get("equipment_id", "340-P-1005 A/B"),
        "fluid": raw.get("fluid", {
            "specific_gravity"   : 1.084,
            "vapor_pressure_psia": 1.02,
            "pump_efficiency"    : 0.686,
        }),
        "head_flow": {
            "rated_flow"          : rated_flow,
            "rated_head"          : h.get("rated_head", 1795.9),
            "por_min_flow"        : por_min,
            "por_max_flow"        : por_max,
            "aor_min_flow"        : aor_min,
            "aor_max_flow"        : aor_max,
            "coefficients"        : coeff_rated,
            "coeff_rated_diameter": coeff_rated,
            "coeff_max_diameter"  : coeff_max,
            "coeff_min_diameter"  : coeff_min,
        },
        "npsh": {
            "npsha"              : npsh.get("npsha", None),
            "vapor_pressure_psia": npsh.get("vapor_pressure_psia", 1.02),
            "specific_gravity"   : npsh.get("specific_gravity", 1.084),
            "npshr_coefficients" : npsh.get("npshr_coefficients",
                                            [-2e-13, 3e-10, -2e-7, 0.0001, -0.0032, 7.8808]),
        },
        "power": {
            "rated_power" : rated_power,
            "coefficients": p.get("coefficients", [1e-12, -2e-9, 1e-6, 0.0002, 0.1081, 292.41]),
        },
    }


def load_compressor_curves(curves_path: str = "curves/compressor_C1001B_curves.json") -> dict:
    if not os.path.exists(curves_path):
        return _default_compressor_curves()
    with open(curves_path) as f:
        return json.load(f)


def _load_fitted_curves(fitted_path: str = "curves/compressor_C1001B_fitted.json") -> dict:
    """Load degree-3 polynomials fitted from actual DCS curve data points."""
    if os.path.exists(fitted_path):
        with open(fitted_path) as f:
            return json.load(f)
    return {}


def _default_pump_curves() -> dict:
    """
    Pump Performance Curves — 340-P-1005 A/B (DMF)
    Sumber : Pump_Performance_mtd.xlsx (PT Pertamina EP Cepu)
    Fluid  : SULFINOL-X (Water, MDEA, Piperazine, Sulfolane)
    Tag DCS: Flow  = Root.U340.F.340FIC1003A.PV  (BPD)
             Ps    = Root.U340.P.340PI1135.PV     (psig)
             Pd    = 340-P-1005 A/B (Local)       (psig)

    ─────────────────────────────────────────────────────────────
    KONVERSI UNIT (dari DCS ke unit kurva):
      Flow (GPM)  = Flow_BPD × 42 / 1440
      Head (ft)   = (Pd_psig − Ps_psig) / (0.4335 × SG)
      BHP  (hp)   = Flow_GPM × (Pd_psig − Ps_psig) / (1715 × η)
      Ps   (psia) = Ps_psig + 14.7
      NPSHa (ft)  = (Ps_psia − Pv_psia) / (0.4335 × SG)
    ─────────────────────────────────────────────────────────────

    PERSAMAAN POLYNOMIAL — np.polyval(coeff, flow_gpm):
      Format koefisien: [x^5, x^4, x^3, x^2, x^1, x^0]

      HEAD vs FLOW (Head ft, Flow GPM):
        Max Diameter  (10.79"):
          y =  4E-12x5 − 6E-9x4 + 3E-7x3 + 0.0003x2 − 0.1651x + 2353.8
        Rated Diameter(10.35"):  ← default untuk cek operasi
          y =  5E-13x5 + 1E-9x4 − 4E-6x3 + 0.0013x2 − 0.2108x + 2146.9
        Min Diameter  (9.65"):
          y = −8E-13x5 + 4E-9x4 − 6E-6x3 + 0.0016x2 − 0.1897x + 1858.3

      POWER vs FLOW (BHP hp, Flow GPM):
        y = 1E-12x5 − 2E-9x4 + 1E-6x3 + 0.0002x2 + 0.1081x + 292.41

      NPSHr vs FLOW (ft, Flow GPM):
        y = −2E-13x5 + 3E-10x4 − 2E-7x3 + 0.0001x2 − 0.0032x + 7.8808

    BATAS OPERATING REGION (GPM absolut dari Pump Raw Data):
      BEP            : 600.2 GPM
      Lower POR      : 420   GPM  (70% BEP)
      Upper POR      : 711.6 GPM (118.6% BEP)
      Min AOR        : 115   GPM  (19.2% BEP)
      Max AOR        : 750   GPM  (125% BEP, estimasi)
    ─────────────────────────────────────────────────────────────
    """
    rated_flow = 600.2   # GPM — BEP (Best Efficiency Point)

    return {
        "equipment_id": "340-P-1005 A/B",

        # ── Fluid Properties ──────────────────────────────────────────
        "fluid": {
            "name"            : "SULFINOL-X",
            "specific_gravity": 1.084,
            "density_kg_m3"   : 1084,
            "viscosity_cp"    : 10.9,
            "vapor_pressure_psia": 1.02,
            "pump_efficiency" : 0.686,   # η — dipakai hitung BHP dari DCS
        },

        # ── Head vs Flow ───────────────────────────────────────────────
        # Batas POR/AOR dalam format PERSEN agar kompatibel dengan
        # check_pump_performance() yang memakai: val/100 * rated_flow
        "head_flow": {
            "rated_flow"   : rated_flow,   # GPM — BEP
            "rated_head"   : 1795.9,       # ft  — head di BEP

            # Batas dalam % — hasil konversi dari GPM absolut
            "por_min_flow" : 420.0 / rated_flow * 100,   # 69.98% → 420 GPM
            "por_max_flow" : 711.6 / rated_flow * 100,   # 118.56% → 711.6 GPM
            "aor_min_flow" : 115.0 / rated_flow * 100,   # 19.16% → 115 GPM
            "aor_max_flow" : 750.0 / rated_flow * 100,   # 124.96% → 750 GPM

            # Koefisien [x^5, x^4, x^3, x^2, x^1, x^0]
            "coeff_max_diameter"  : [ 4e-12, -6e-9,  3e-7,  0.0003, -0.1651, 2353.8],
            "coeff_rated_diameter": [ 5e-13,  1e-9, -4e-6,  0.0013, -0.2108, 2146.9],
            "coeff_min_diameter"  : [-8e-13,  4e-9, -6e-6,  0.0016, -0.1897, 1858.3],

            # "coefficients" = rated diameter — dipakai check_pump_performance()
            "coefficients"        : [ 5e-13,  1e-9, -4e-6,  0.0013, -0.2108, 2146.9],
        },

        # ── NPSH ─────────────────────────────────────────────────────
        # NPSHa = (Ps_psia − Pv_psia) / (0.4335 × SG)
        # Ps_psia = Ps_psig + 14.7
        "npsh": {
            "npsha"              : None,    # dihitung dinamis dari DCS, di-set di app.py
            "vapor_pressure_psia": 1.02,
            "specific_gravity"   : 1.084,
            # NPSHr polynomial [x^5, x^4, x^3, x^2, x^1, x^0]
            "npshr_coefficients" : [-2e-13, 3e-10, -2e-7, 0.0001, -0.0032, 7.8808],
        },

        # ── Power vs Flow ─────────────────────────────────────────────
        # BHP (hp) = Flow_GPM × (Pd − Ps) / (1715 × η)
        "power": {
            "rated_power" : 500.0,   # hp — rated motor power
            # Koefisien [x^5, x^4, x^3, x^2, x^1, x^0]
            "coefficients": [1e-12, -2e-9, 1e-6, 0.0002, 0.1081, 292.41],
        },
    }


def _default_compressor_curves() -> dict:
    """Default curves for C-1001B (Donggi Compressor BCL305) — from actual DCS data."""
    return {
        "equipment_id": "C-1001B",
        "equipment_name": "Donggi Compressor - BCL305",
        "speed_lines_rpm": {
            "A": 13068, "B": 12446, "C": 11201,
            "D": 9957,  "E": 8712,  "average": 10827.2
        },
        "poly_head": {
            "unit": "kJ/kg",
            "speed_A": [-0.000004, 0.0012, -0.1494, 8.9335, -265.78, 3289.9],
            "speed_B": [-0.000002, 0.0006, -0.0692, 3.6827, -96.807, 1135.6],
            "speed_C": [-0.000002, 0.0004, -0.0393, 1.8581, -43.538, 508.57],
            "speed_D": [0.0, -0.000002, 0.0012, -0.1012, 2.9854, 52.543],
            "speed_E": [0.000004, -0.0008, 0.0496, -1.6027, 24.924, -84.818],
            "surge_line": [-0.00003, 0.0049, -0.3786, 14.385, -265.28, 1937.1],
            "protection_line": [-0.00001, 0.0025, -0.2133, 9.0767, -186.28, 1512.4],
            "stonewall": [0.0, 0.0, 0.0, 0.0, 1.5909, -31.846]
        },
        "pressure_ratio": {
            "speed_A": [0.0, 0.00002, -0.0025, 0.1523, -4.5503, 56.578],
            "speed_B": [0.0, 0.00001, -0.0011, 0.0603, -1.6299, 19.836],
            "speed_C": [0.0, 0.0, 0.00001, -0.0017, 0.0672, 1.1336],
            "speed_D": [0.0, 0.0, 0.00002, -0.0019, 0.0615, 1.0898]
        },
        "shaft_power": {
            "unit": "kW",
            "rated_power": 1280.0,
            "speed_A": [-0.00006, 0.0174, -2.1307, 129.76, -3891.2, 47921.0],
            "speed_B": [-0.00003, 0.0094, -1.0706, 60.068, -1634.7, 18878.0],
            "speed_C": [-0.00002, 0.0044, -0.4536, 22.701, -532.98, 5833.0],
            "speed_D": [0.00002, -0.0034, 0.2911, -12.383, 284.07, -1957.9],
            "speed_E": [0.00002, -0.0038, 0.2605, -9.0638, 174.88, -940.75]
        },
        "flow_pressure_ratio": {
            "surge_flow": 27.09,
            "protection_flow": 29.54,
            "rated_flow": 55.87,
            "rated_pressure_ratio": 1.767
        },
        "operating_stats": {
            "flow_p10": 46.86, "flow_p25": 50.21,
            "flow_p75": 54.69, "flow_p90": 55.03,
            "shaft_power_mean": 1243.0, "shaft_power_p99": 1278.2,
            "poly_head_mean": 79.46
        }
    }


# -------------------------------------------------
# PUMP PERFORMANCE CHECK
# -------------------------------------------------

def check_pump_performance(equipment_id, flow, head, power, npsha, curves) -> PumpStatus:
    """
    Evaluasi titik operasi pompa terhadap kurva pabrikan.
    Rules persis dari Pump_Performance_mtd.xlsx — sheet Summary.

    RULES HEAD vs FLOW:
      Qactual > Q_max_diameter        → High Flow / Pump Overload
      Qactual < Q_min_diameter        → Low Flow / Risk of Recirculation
      Q_lower_POR ≤ Qactual ≤ Q_upper_POR → POR (Best Efficiency)
      Q_min_AOR ≤ Qactual < Q_lower_POR   → AOR Low Flow
      Qactual < Q_min_AOR             → Overheating, seal damage, recirculation
      Qactual > Q_max_AOR             → Motor overload, impeller damage, cavitation

    RULES POWER vs FLOW:
      P_actual > 110% P_curve         → Overload
      P_actual < 90% P_curve          → Possible Flow Measurement Error

    RULES NPSHa vs NPSHr:
      NPSHa > NPSHr                   → Pump aman
      NPSHa > 1.3 × NPSHr            → Risiko kavitasi (berisiko)
      NPSHa ≤ NPSHr                   → Kavitasi terjadi

    Parameters
    ----------
    flow   : float  Flow aktual dalam GPM
                    Konversi dari BPD: flow_gpm = flow_bpd * 42 / 1440
    head   : float  Differential head dalam ft
                    Dari DCS: (Pd_psig - Ps_psig) / (0.4335 * SG)
    power  : float  Brake Horsepower dalam hp
                    Dari DCS: (flow_gpm * (Pd_psig - Ps_psig)) / (1715 * efficiency)
    npsha  : float  NPSH available dalam ft
                    Dari DCS: (Ps_psia - Pv_psia) / (0.4335 * SG)
                    dengan Ps_psia = Ps_psig + 14.7
    """
    alerts = []
    hf         = curves["head_flow"]
    rated_flow = hf["rated_flow"]   # 600.2 GPM (BEP)
    sg         = curves["npsh"].get("specific_gravity", 1.084)

    # ── Batas operating region (GPM absolut) ─────────────────────────
    por_min = hf["por_min_flow"] / 100 * rated_flow   # 420.0 GPM
    por_max = hf["por_max_flow"] / 100 * rated_flow   # 711.6 GPM
    aor_min = hf["aor_min_flow"] / 100 * rated_flow   # 115.0 GPM
    aor_max = hf["aor_max_flow"] / 100 * rated_flow   # 750.0 GPM

    # ── Head pada tiap kurva diameter di flow aktual ──────────────────
    coeff_rated = hf["coefficients"]   # Rated diameter — referensi utama
    coeff_max   = hf.get("coeff_max_diameter",   coeff_rated)
    coeff_min   = hf.get("coeff_min_diameter",   coeff_rated)

    head_rated_curve = np.polyval(coeff_rated, flow)
    head_max_curve   = np.polyval(coeff_max,   flow)
    head_min_curve   = np.polyval(coeff_min,   flow)

    # Q_max_diameter dan Q_min_diameter: flow di mana head curve = 0
    # Approx: pakai aor_max dan aor_min sebagai proxy (sesuai Excel rules)
    q_max_diameter = aor_max   # 750 GPM — batas envelope max
    q_min_diameter = aor_min   # 115 GPM — batas envelope min

    # ── NPSHr dari kurva polynomial ───────────────────────────────────
    npshr = np.polyval(curves["npsh"]["npshr_coefficients"], flow)
    npshr = max(npshr, 0.0)

    # Tangani npsha=None (belum ada data DCS)
    if npsha is None:
        npsha = 9999.0

    # ── Power dari kurva ──────────────────────────────────────────────
    rated_power    = curves["power"]["rated_power"]
    expected_power = np.polyval(curves["power"]["coefficients"], flow)
    expected_power = max(expected_power, 1.0)

    # ════════════════════════════════════════════════════════════════
    # ZONE CLASSIFICATION — sesuai urutan prioritas rules Excel
    # ════════════════════════════════════════════════════════════════

    # ── 1. NPSH CHECK (prioritas tertinggi) ──────────────────────────
    if npsha <= npshr:
        # NPSHa ≤ NPSHr → Kavitasi terjadi
        zone = PumpZone.CAVITATION_RISK
        alerts.append(
            f"⛔ KRITIS: NPSHa ({npsha:.1f} ft) ≤ NPSHr ({npshr:.1f} ft) "
            f"— Terjadi Kavitasi pada pump!"
        )

    elif npsha <= 1.3 * npshr:
        # NPSHa > NPSHr tapi < 1.3×NPSHr → berisiko kavitasi
        zone = PumpZone.AOR
        alerts.append(
            f"⚠️ PERINGATAN: NPSHa ({npsha:.1f} ft) < 1.3×NPSHr ({1.3*npshr:.1f} ft) "
            f"— Pump berisiko terjadi kavitasi"
        )

    # ── 2. FLOW vs ENVELOPE KURVA ────────────────────────────────────
    elif flow > q_max_diameter:
        # Qactual > Q_max_diameter → High Flow / Pump Overload
        zone = PumpZone.OVERLOAD
        alerts.append(
            f"⛔ KRITIS: Flow ({flow:.1f} GPM) > Q_max_diameter ({q_max_diameter:.1f} GPM) "
            f"— High Flow / Pump Overload"
        )

    elif flow < q_min_diameter:
        # Qactual < Q_min_diameter → Low Flow / Risk of Recirculation
        zone = PumpZone.MINIMUM_FLOW
        alerts.append(
            f"⛔ KRITIS: Flow ({flow:.1f} GPM) < Q_min_diameter ({q_min_diameter:.1f} GPM) "
            f"— Low Flow / Risk of Recirculation, overheating, seal damage"
        )

    # ── 3. POR CHECK ─────────────────────────────────────────────────
    elif por_min <= flow <= por_max:
        # Qactual ≥ Q_lower_POR DAN Qactual ≤ Q_upper_POR → POR
        zone = PumpZone.POR

    # ── 4. AOR LOW CHECK ─────────────────────────────────────────────
    elif aor_min <= flow < por_min:
        # Q_min_AOR ≤ Qactual < Q_lower_POR → AOR Low Flow
        zone = PumpZone.AOR
        alerts.append(
            f"⚠️ PERINGATAN: Flow ({flow:.1f} GPM) di bawah POR ({por_min:.0f} GPM) "
            f"— Allowable Operating Region - Low Flow"
        )

    elif por_max < flow <= aor_max:
        # Q_upper_POR < Qactual ≤ Q_max_AOR → AOR High Flow
        zone = PumpZone.AOR
        alerts.append(
            f"⚠️ PERINGATAN: Flow ({flow:.1f} GPM) di atas POR ({por_max:.0f} GPM) "
            f"— Allowable Operating Region - High Flow"
        )

    else:
        zone = PumpZone.AOR
        alerts.append(f"⚠️ PERINGATAN: Flow ({flow:.1f} GPM) di luar POR — cek kondisi operasi")

    # ── 5. POWER CHECK ────────────────────────────────────────────────
    # P_actual > 110% P_curve → Overload (override zone)
    if power > 1.10 * expected_power:
        zone = PumpZone.OVERLOAD
        alerts.append(
            f"⛔ KRITIS: BHP aktual ({power:.1f} hp) > 110% kurva "
            f"({expected_power:.1f} hp) — Overload"
        )
    # P_actual < 90% P_curve → Possible Flow Measurement Error
    elif power < 0.90 * expected_power:
        alerts.append(
            f"⚠️ INFO: BHP aktual ({power:.1f} hp) < 90% kurva "
            f"({expected_power:.1f} hp) — Possible Flow Measurement Error"
        )

    # ── 6. EFISIENSI POMPA ────────────────────────────────────────────
    # Formula imperial: η = (Q × H × SG) / (3960 × BHP) × 100%
    efficiency = (flow * head * sg) / (3960.0 * power) * 100 if power > 0 else 0.0

    # ── 7. STATUS NORMAL ─────────────────────────────────────────────
    if not alerts:
        alerts.append(
            f"✅ Normal: Pump aman — NPSHa ({npsha:.1f} ft) > NPSHr ({npshr:.1f} ft), "
            f"beroperasi dalam zona POR "
            f"(flow={flow:.1f} GPM, head={head:.0f} ft, η={efficiency:.1f}%)"
        )

    return PumpStatus(
        equipment_id  = equipment_id,
        flow          = flow,
        head          = head,
        power         = power,
        npsha         = npsha,
        zone          = zone,
        efficiency    = round(efficiency, 2),
        alert_messages= alerts,
    )


# -------------------------------------------------
# COMPRESSOR PERFORMANCE CHECK -- 8 ZONE (HYBRID)
# -------------------------------------------------

def check_compressor_performance(equipment_id: str,
                                  inlet_flow: float,
                                  pressure_ratio: float,
                                  poly_head: float,
                                  shaft_power: float,
                                  poly_efficiency: float,
                                  speed_rpm: float,
                                  curves: dict) -> CompressorStatus:
    """
    Evaluate C-1001B operating point with 8-zone classification.

    Z1/Z2 use flow-based boundaries (reliable across all conditions).
    Z3-Z8 compare actual poly_head against fitted speed-line curves
    loaded from compressor_C1001B_fitted.json (degree-3 polynomials
    fitted directly from raw DCS chart data points).
    """
    alerts = []

    fpr = curves["flow_pressure_ratio"]
    surge_flow      = fpr.get("surge_flow", 27.09)
    protection_flow = fpr.get("protection_flow", 29.54)
    rated_power     = curves["shaft_power"].get("rated_power", 1280.0)

    # Surge margin (flow-based, %)
    surge_margin_pct = (inlet_flow - surge_flow) / surge_flow * 100 if surge_flow > 0 else 100.0

    # Load fitted polynomials from actual data points
    fitted = _load_fitted_curves()

    def hp_at(key: str, q: float) -> float:
        if key in fitted:
            return float(np.polyval(fitted[key]["coeff"], q))
        return float("nan")

    hp_B = hp_at("speed_B", inlet_flow)
    hp_C = hp_at("speed_C", inlet_flow)
    hp_D = hp_at("speed_D", inlet_flow)
    hp_E = hp_at("speed_E", inlet_flow)

    # --- Zone classification ---
    if inlet_flow <= surge_flow:
        zone = CompressorZone.SURGE
        alerts.append(f"BAHAYA SURGE! Flow={inlet_flow:.2f} MMSCFD <= Surge Line={surge_flow:.2f} MMSCFD")
        alerts.append("SEGERA: Buka Anti-Surge Valve! Kurangi beban! Hubungi engineer lapangan!")

    elif inlet_flow <= protection_flow:
        zone = CompressorZone.PROTECTION
        alerts.append(f"Zona Proteksi Anti-Surge. Flow={inlet_flow:.2f} MMSCFD | Surge margin={surge_margin_pct:.1f}%")
        alerts.append("Pantau tren flow & polytropic head. Pastikan anti-surge control aktif.")

    elif not np.isnan(hp_B) and poly_head >= hp_B:
        zone = CompressorZone.OPTIMAL
        alerts.append(f"Zona Operasi Optimal (antara SpeedB & Protection Line). Hp={poly_head:.2f} kJ/kg")

    elif not np.isnan(hp_C) and poly_head >= hp_C:
        zone = CompressorZone.NORMAL
        alerts.append(f"Zona Operasi Normal (antara SpeedC & SpeedB). Hp={poly_head:.2f} kJ/kg")

    elif not np.isnan(hp_D) and poly_head >= hp_D:
        zone = CompressorZone.LOW_SPEED
        alerts.append(f"Zona Kecepatan Rendah (antara SpeedD & SpeedC). Hp={poly_head:.2f} kJ/kg | Speed={speed_rpm:.0f} RPM")
        alerts.append("Efisiensi menurun. Pantau kondisi operasi dan evaluasi set-point speed.")

    elif not np.isnan(hp_E) and poly_head >= hp_E:
        zone = CompressorZone.MIN_SPEED
        alerts.append(f"Zona Kecepatan Minimum (antara SpeedE & SpeedD). Hp={poly_head:.2f} kJ/kg | Speed={speed_rpm:.0f} RPM")
        alerts.append("Operasi dekat batas minimum. Pertimbangkan peningkatan speed atau intervensi operasi.")

    elif poly_head > 10:
        zone = CompressorZone.BELOW_ENVELOPE
        alerts.append(f"Di Bawah Envelope Operasi! Hp={poly_head:.2f} kJ/kg < SpeedE={hp_E:.2f} kJ/kg")
        alerts.append("Titik operasi di luar peta performa normal. Periksa kondisi inlet dan speed kompresor.")

    else:
        zone = CompressorZone.STONEWALL
        alerts.append(f"KRITIS: Kompresor di Zona Stonewall/Choke! Hp={poly_head:.2f} kJ/kg")
        alerts.append("Tindakan: Kurangi aliran, periksa kondisi downstream, hubungi engineer segera.")

    # Shaft power overload override
    if shaft_power > 1.1 * rated_power:
        zone = CompressorZone.SURGE
        alerts.append(f"KRITIS: Shaft Power ({shaft_power:.0f} kW) melebihi 110% rated ({rated_power:.0f} kW)")

    # Low efficiency warning
    poly_eff_mean = 81.38
    if poly_efficiency < poly_eff_mean - 1.0:
        alerts.append(f"Efisiensi polirtopik rendah: {poly_efficiency:.2f}% (rata-rata normal: {poly_eff_mean:.2f}%)")

    if not alerts:
        alerts.append(f"Normal: Kompresor beroperasi dalam zona {zone.value}")

    return CompressorStatus(
        equipment_id=equipment_id,
        inlet_flow=inlet_flow,
        pressure_ratio=pressure_ratio,
        polytropic_head=poly_head,
        shaft_power=shaft_power,
        poly_efficiency=poly_efficiency,
        speed_rpm=speed_rpm,
        zone=zone,
        surge_margin_pct=surge_margin_pct,
        alert_messages=alerts
    )
