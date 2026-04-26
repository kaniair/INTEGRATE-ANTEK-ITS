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
    if not os.path.exists(curves_path):
        return _default_pump_curves()
    with open(curves_path) as f:
        return json.load(f)


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
    return {
        "equipment_id": "P-1001A",
        "head_flow": {
            "por_min_flow": 60.0,
            "por_max_flow": 110.0,
            "aor_min_flow": 40.0,
            "aor_max_flow": 120.0,
            "rated_flow": 371.0,
            "rated_head": 1954.0,
            "coefficients": [-0.0012, -0.1, 1954.0]
        },
        "npsh": {
            "npsha": 25.0,
            "npshr_coefficients": [0.0001, 0.05, 3.0]
        },
        "power": {
            "rated_power": 450.0,
            "coefficients": [0.001, 0.5, 50.0]
        }
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
    alerts = []
    hf = curves["head_flow"]
    rated_flow = hf["rated_flow"]
    por_min = hf["por_min_flow"] / 100 * rated_flow
    por_max = hf["por_max_flow"] / 100 * rated_flow
    aor_min = hf["aor_min_flow"] / 100 * rated_flow
    aor_max = hf["aor_max_flow"] / 100 * rated_flow

    coeff = hf["coefficients"]
    expected_head = np.polyval(coeff, flow)

    npsh_coeff = curves["npsh"]["npshr_coefficients"]
    npshr = np.polyval(npsh_coeff, flow)

    if npsha <= npshr:
        zone = PumpZone.CAVITATION_RISK
        alerts.append(f"KRITIS: NPSHa ({npsha:.1f}) <= NPSHr ({npshr:.1f}) -- Kavitasi aktif!")
    elif npsha <= 1.3 * npshr:
        zone = PumpZone.AOR
        alerts.append(f"PERINGATAN: NPSHa ({npsha:.1f}) mendekati NPSHr -- Risiko kavitasi")
    elif flow < aor_min:
        zone = PumpZone.MINIMUM_FLOW
        alerts.append(f"KRITIS: Flow ({flow:.1f}) di bawah minimum flow AOR ({aor_min:.1f})")
    elif flow > aor_max:
        zone = PumpZone.AOR
        alerts.append(f"PERINGATAN: Flow ({flow:.1f}) melebihi AOR maksimum ({aor_max:.1f})")
    elif por_min <= flow <= por_max:
        zone = PumpZone.POR
    else:
        zone = PumpZone.AOR
        alerts.append(f"PERINGATAN: Pompa beroperasi di AOR (flow={flow:.1f})")

    rated_power = curves["power"]["rated_power"]
    if power > 1.2 * rated_power:
        zone = PumpZone.OVERLOAD
        alerts.append(f"KRITIS: Daya motor ({power:.1f} kW) melebihi 120% rated power ({rated_power:.1f} kW)")

    efficiency = (flow * head * 9.81 * 1000) / (power * 3600 * 1000) * 100 if power > 0 else 0

    if not alerts:
        alerts.append("Normal: Pompa beroperasi dalam zona POR")

    return PumpStatus(equipment_id=equipment_id, flow=flow, head=head,
                      power=power, npsha=npsha, zone=zone,
                      efficiency=efficiency, alert_messages=alerts)


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
