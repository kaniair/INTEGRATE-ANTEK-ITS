"""
INTEGRATE - Performance Curve Engine
Rule-based engine for real-time pump & compressor performance curve monitoring.
Standards: ASME PTC 8.2, API 610 (pump), ASME PTC 10, API 617 (compressor).
"""

import numpy as np
import json
import os
from dataclasses import dataclass
from enum import Enum

# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────


class PumpZone(Enum):
    POR = "POR"  # Preferred Operating Region ✅
    AOR = "AOR"  # Allowable Operating Region ⚠️
    CAVITATION_RISK = "Cavitation"  # NPSHa margin too low 🔴
    MINIMUM_FLOW = "Min Flow"  # Below minimum flow 🔴
    OVERLOAD = "Overload"  # Power overload 🔴


class CompressorZone(Enum):
    OPTIMAL = "Optimal"  # Safe operating zone ✅
    PROTECTION = "Protection Line"  # Warning zone ⚠️
    SURGE = "Surge Zone"  # Dangerous ─ immediate action 🔴
    OVERLOAD = "Overload"  # Shaft power exceeded 🔴


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
    surge_margin: float
    zone: CompressorZone
    alert_messages: list


# ─────────────────────────────────────────────
# CURVE LOADING
# ─────────────────────────────────────────────


def load_pump_curves(curves_path: str = "curves/pump_curves.json") -> dict:
    """Load polynomial coefficients for pump curves."""
    if not os.path.exists(curves_path):
        # Return default sample curves if file not found
        return _default_pump_curves()

    with open(curves_path) as f:
        return json.load(f)


def load_compressor_curves(curves_path: str = "curves/compressor_curves.json") -> dict:
    """Load polynomial coefficients for compressor curves."""
    if not os.path.exists(curves_path):
        return _default_compressor_curves()

    with open(curves_path) as f:
        return json.load(f)


def _default_pump_curves() -> dict:
    """
    Default sample curves — replace with actual datasheet coefficients
    extracted using WebPlotDigitizer.
    """
    return {
        "equipment_id": "P-1001A",
        "head_flow": {
            "por_min_flow": 60.0,  # % of rated flow
            "por_max_flow": 110.0,
            "aor_min_flow": 40.0,
            "aor_max_flow": 120.0,
            "rated_flow": 371.0,  # kBPD or m³/h
            "rated_head": 1954.0,  # ft or m
            "coefficients": [-0.0012, -0.1, 1954.0],  # ax² + bx + c
        },
        "npsh": {
            "npsha": 25.0,  # ft — available NPSH
            "npshr_coefficients": [0.0001, 0.05, 3.0],  # NPSHr curve
        },
        "power": {"rated_power": 450.0, "coefficients": [0.001, 0.5, 50.0]},  # kW
    }


def _default_compressor_curves() -> dict:
    """
    Default sample compressor curves — replace with actual C-1001A/B
    datasheet values from KP DMF.
    """
    return {
        "equipment_id": "C-1001A",
        "flow_pressure_ratio": {
            "surge_flow": 25.0,  # MMSCFD — surge line flow
            "protection_margin": 0.10,  # 10% above surge line
            "rated_flow": 55.4,  # MMSCFD
            "rated_pressure_ratio": 1.82,
            "speed_lines": {  # RPM → [flow_points, pr_points]
                "100%": {"flow": [30, 45, 55, 65], "pr": [2.1, 1.95, 1.82, 1.65]},
                "95%": {"flow": [28, 42, 52, 60], "pr": [1.95, 1.80, 1.68, 1.52]},
                "90%": {"flow": [25, 38, 48, 55], "pr": [1.80, 1.66, 1.55, 1.40]},
            },
        },
        "flow_polytropic_head": {
            "rated_flow": 55.4,
            "rated_hp": 45.2,  # kJ/kg
            "coefficients": [-0.02, 2.5, 30.0],
        },
        "shaft_power": {
            "rated_power": 1312.0,  # kW
            "coefficients": [0.3, 10.0, 200.0],
        },
    }


# ─────────────────────────────────────────────
# PUMP PERFORMANCE CHECK
# ─────────────────────────────────────────────


def check_pump_performance(
    equipment_id: str,
    flow: float,
    head: float,
    power: float,
    npsha: float,
    curves: dict,
) -> PumpStatus:
    """
    Evaluate pump operating point against manufacturer curves.
    Returns PumpStatus with zone classification and alerts.

    Parameters match API 610 / ASME PTC 8.2 standards.
    """
    alerts = []

    hf = curves["head_flow"]
    rated_flow = hf["rated_flow"]
    por_min = hf["por_min_flow"] / 100 * rated_flow
    por_max = hf["por_max_flow"] / 100 * rated_flow
    aor_min = hf["aor_min_flow"] / 100 * rated_flow
    aor_max = hf["aor_max_flow"] / 100 * rated_flow

    # ── Head vs Flow: calculate expected head at current flow ──
    coeff = hf["coefficients"]
    expected_head = np.polyval(coeff, flow)
    head_deviation = abs(head - expected_head) / expected_head * 100

    # ── NPSHa check (ANSI/HI 9.6.1-2014) ──
    npsh_coeff = curves["npsh"]["npshr_coefficients"]
    npshr = np.polyval(npsh_coeff, flow)

    if npsha <= npshr:
        zone = PumpZone.CAVITATION_RISK
        alerts.append(
            f"⛔ KRITIS: NPSHa ({npsha:.1f}) ≤ NPSHr ({npshr:.1f}) — Kavitasi aktif!"
        )
    elif npsha <= 1.3 * npshr:
        zone = PumpZone.AOR
        alerts.append(
            f"⚠️ PERINGATAN: NPSHa ({npsha:.1f}) mendekati NPSHr ({npshr:.1f}) — Risiko kavitasi"
        )
    elif flow < aor_min:
        zone = PumpZone.MINIMUM_FLOW
        alerts.append(
            f"⛔ KRITIS: Flow ({flow:.1f}) di bawah minimum flow AOR ({aor_min:.1f})"
        )
    elif flow > aor_max:
        zone = PumpZone.AOR
        alerts.append(
            f"⚠️ PERINGATAN: Flow ({flow:.1f}) melebihi AOR maksimum ({aor_max:.1f})"
        )
    elif por_min <= flow <= por_max:
        zone = PumpZone.POR
    else:
        zone = PumpZone.AOR
        alerts.append(f"⚠️ PERINGATAN: Pompa beroperasi di AOR (flow={flow:.1f})")

    # ── Power check ──
    power_coeff = curves["power"]["coefficients"]
    rated_power = curves["power"]["rated_power"]
    expected_power = np.polyval(power_coeff, flow)
    if power > 1.2 * rated_power:
        zone = PumpZone.OVERLOAD
        alerts.append(
            f"⛔ KRITIS: Daya motor ({power:.1f} kW) melebihi 120% rated power ({rated_power:.1f} kW)"
        )

    # ── Efficiency ──
    efficiency = (
        (flow * head * 9.81 * 1000) / (power * 3600 * 1000) * 100 if power > 0 else 0
    )

    if not alerts:
        alerts.append(f"✅ Normal: Pompa beroperasi dalam zona POR")

    return PumpStatus(
        equipment_id=equipment_id,
        flow=flow,
        head=head,
        power=power,
        npsha=npsha,
        zone=zone,
        efficiency=efficiency,
        alert_messages=alerts,
    )


# ─────────────────────────────────────────────
# COMPRESSOR PERFORMANCE CHECK
# ─────────────────────────────────────────────


def check_compressor_performance(
    equipment_id: str,
    inlet_flow: float,
    pressure_ratio: float,
    polytropic_head: float,
    shaft_power: float,
    curves: dict,
) -> CompressorStatus:
    """
    Evaluate compressor operating point against manufacturer curves.
    Includes Surge Margin Index calculation.
    Standards: API 617, ASME PTC 10.
    """
    alerts = []

    fpr = curves["flow_pressure_ratio"]
    surge_flow = fpr["surge_flow"]
    protection_margin = fpr["protection_margin"]
    protection_flow = surge_flow * (1 + protection_margin)

    # ── Surge Margin Index ──
    # SMI = (actual_flow - surge_flow) / surge_flow × 100%
    surge_margin = (
        (inlet_flow - surge_flow) / surge_flow * 100 if surge_flow > 0 else 100
    )

    # ── Zone classification ──
    if inlet_flow <= surge_flow:
        zone = CompressorZone.SURGE
        alerts.append(
            f"⛔ BAHAYA: Kompresor dalam Surge Zone! Flow={inlet_flow:.1f} ≤ Surge Flow={surge_flow:.1f} MMSCFD"
        )
        alerts.append(
            "🚨 Tindakan segera: Hubungi engineer lapangan, buka Anti-Surge Valve!"
        )
    elif inlet_flow <= protection_flow:
        zone = CompressorZone.PROTECTION
        alerts.append(
            f"⚠️ PERINGATAN: Mendekati surge line. Surge Margin={surge_margin:.1f}% (minimum 10%)"
        )
        alerts.append("Pantau tren flow dan siapkan intervensi operasi")
    else:
        zone = CompressorZone.OPTIMAL
        alerts.append(
            f"✅ Normal: Kompresor di zona optimal. Surge Margin={surge_margin:.1f}%"
        )

    # ── Shaft power check ──
    rated_power = curves["shaft_power"]["rated_power"]
    if shaft_power > 1.1 * rated_power:
        zone = CompressorZone.OVERLOAD
        alerts.append(
            f"⛔ KRITIS: Shaft Power ({shaft_power:.0f} kW) melebihi 110% rated ({rated_power:.0f} kW)"
        )

    return CompressorStatus(
        equipment_id=equipment_id,
        inlet_flow=inlet_flow,
        pressure_ratio=pressure_ratio,
        polytropic_head=polytropic_head,
        shaft_power=shaft_power,
        surge_margin=surge_margin,
        zone=zone,
        alert_messages=alerts,
    )
