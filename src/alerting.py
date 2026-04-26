"""
INTEGRATE - Alerting Module
Automated email notification via smtplib when anomaly detected.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_anomaly_alert(equipment_id: str, equipment_type: str,
                       anomaly_class: str, mae_value: float,
                       threshold: float, parameters: dict,
                       recommended_action: str):
    """
    Send automated email alert when anomaly is detected.
    
    Parameters
    ----------
    equipment_id : str
        Equipment tag (e.g., P-9027B)
    equipment_type : str
        'pump' or 'compressor'
    anomaly_class : str
        ML classification result
    mae_value : float
        Reconstruction error
    threshold : float
        Anomaly threshold
    parameters : dict
        Current operational parameters
    recommended_action : str
        Recommended maintenance action
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    recipients_str = os.getenv("ALERT_RECIPIENTS", "")

    if not smtp_user or not smtp_pass:
        print("[Alert] Email credentials not configured. Skipping email alert.")
        print(f"[Alert] ANOMALY DETECTED: {equipment_id} | Class: {anomaly_class} | MAE: {mae_value:.4f}")
        return False

    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    if not recipients:
        print("[Alert] No recipients configured.")
        return False

    # ── Build email ──
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[INTEGRATE] {equipment_type.upper()} Anomali — {equipment_id} — {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)

    # Plain text version
    text_body = f"""
INTEGRATE MONITORING SYSTEM — NOTIFIKASI ANOMALI

Equipment  : {equipment_id}
Tipe       : {equipment_type.upper()}
Waktu      : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Klasifikasi: {anomaly_class}
MAE        : {mae_value:.4f} (Threshold: {threshold:.4f})

PARAMETER SAAT INI:
{chr(10).join([f'  {k}: {v}' for k, v in parameters.items()])}

REKOMENDASI TINDAKAN:
{recommended_action}

---
Sistem INTEGRATE | PT Pertamina EP Cepu
Pesan ini dikirim otomatis. Jangan balas email ini.
"""

    # HTML version
    params_rows = "".join([
        f"<tr><td style='padding:4px 8px;font-weight:bold;'>{k}</td><td style='padding:4px 8px;'>{v}</td></tr>"
        for k, v in parameters.items()
    ])

    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
    <div style="background:#1a1a2e;color:white;padding:16px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0;">⚠️ INTEGRATE — Notifikasi Anomali</h2>
    </div>
    <div style="border:1px solid #ddd;padding:16px;">
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:6px;width:40%;color:#666;">Equipment</td>
                <td style="padding:6px;font-weight:bold;font-size:18px;">{equipment_id}</td></tr>
            <tr style="background:#f9f9f9;"><td style="padding:6px;color:#666;">Tipe</td>
                <td style="padding:6px;">{equipment_type.upper()}</td></tr>
            <tr><td style="padding:6px;color:#666;">Waktu Deteksi</td>
                <td style="padding:6px;">{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</td></tr>
            <tr style="background:#f9f9f9;"><td style="padding:6px;color:#666;">Klasifikasi ML</td>
                <td style="padding:6px;font-weight:bold;color:#e74c3c;">{anomaly_class}</td></tr>
            <tr><td style="padding:6px;color:#666;">MAE Value</td>
                <td style="padding:6px;">{mae_value:.4f} (Threshold: {threshold:.4f})</td></tr>
        </table>

        <h3 style="color:#2c3e50;border-bottom:1px solid #eee;padding-bottom:8px;">
            Parameter Operasi Saat Ini
        </h3>
        <table style="width:100%;border-collapse:collapse;background:#f8f8f8;">
            {params_rows}
        </table>

        <div style="background:#ffeaa7;border-left:4px solid #fdcb6e;padding:12px;margin-top:16px;border-radius:4px;">
            <strong>📋 Rekomendasi Tindakan:</strong><br/>
            {recommended_action}
        </div>

        <p style="color:#999;font-size:12px;margin-top:24px;">
            Pesan ini dikirim otomatis oleh sistem INTEGRATE.<br/>
            PT Pertamina EP Cepu — {datetime.now().year}
        </p>
    </div>
</body>
</html>
"""

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # ── Send ──
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())
        print(f"[Alert] Email sent to: {', '.join(recipients)}")
        return True
    except Exception as e:
        print(f"[Alert] Failed to send email: {e}")
        return False


def get_recommended_action(anomaly_class: str, equipment_type: str) -> str:
    """Return maintenance recommendation based on anomaly class / zone."""
    actions = {
        "pump": {
            "Equipment": "Cek mechanical seal dan bearing. Lakukan inspeksi visual segera. Koordinasi dengan maintenance.",
            "Proses":    "Periksa kondisi proses upstream/downstream. Cek valve posisi dan flow control.",
            "Startup":   "Monitor parameter startup. Pastikan sequence startup sesuai SOP.",
            "Nominasi":  "Evaluasi operating point terhadap kurva performa. Pertimbangkan penyesuaian laju produksi.",
            "Normal":    "Tidak ada tindakan diperlukan. Lanjutkan pemantauan rutin.",
            "Unknown":   "Lakukan inspeksi manual dan konsultasi dengan engineer lapangan."
        },
        "compressor": {
            # Zone-based classifications for C-1001B
            "Z1 - Surge Zone":
                "🚨 TINDAKAN SEGERA: (1) Buka Anti-Surge Valve (ASV) penuh. "
                "(2) Kurangi laju produksi/beban kompresor. "
                "(3) Hubungi engineer lapangan & control room. "
                "(4) Siapkan prosedur emergency shutdown jika diperlukan.",

            "Z2 - Protection Zone":
                "⚠️ WASPADA: (1) Monitor surge margin secara real-time. "
                "(2) Pastikan Anti-Surge Control dalam mode AUTO. "
                "(3) Hindari penurunan flow lebih lanjut. "
                "(4) Siapkan operator standby di lapangan.",

            "Z3 - Zona Operasi Optimal":
                "✅ Kompresor beroperasi optimal antara SpeedB dan Protection Line. "
                "Pertahankan kondisi operasi saat ini. Lanjutkan monitoring rutin.",

            "Z4 - Zona Operasi Normal":
                "✅ Kompresor beroperasi normal (antara SpeedC dan SpeedB). "
                "Kondisi aman. Pantau tren parameter secara berkala.",

            "Z5 - Zona Kecepatan Rendah":
                "⚠️ Kompresor beroperasi di zona kecepatan rendah (antara SpeedD dan SpeedC). "
                "(1) Evaluasi set-point speed kompresor. "
                "(2) Cek kondisi governor / speed controller. "
                "(3) Monitor efisiensi polirtopik — pertimbangkan optimasi.",

            "Z6 - Zona Kecepatan Minimum":
                "⚠️ PERHATIAN: Kompresor mendekati batas minimum speed (antara SpeedE dan SpeedD). "
                "(1) Segera evaluasi speed controller. "
                "(2) Koordinasi dengan production engineer untuk penyesuaian beban. "
                "(3) Lakukan inspeksi mechanical lebih awal.",

            "Z7 - Di Bawah Envelope":
                "⛔ KRITIS: Titik operasi di luar envelope performa normal (di bawah SpeedE). "
                "(1) Periksa kondisi inlet gas (tekanan, temperatur, komposisi). "
                "(2) Cek instrument DCS — kemungkinan faulty sensor. "
                "(3) Hubungi engineer & lakukan verifikasi manual.",

            "Z8 - Stonewall / Choke":
                "⛔ KRITIS: Kompresor mendekati kondisi Stonewall/Choke. "
                "(1) Kurangi flow rate segera. "
                "(2) Periksa kondisi downstream (valve, line obstruction). "
                "(3) Koordinasi dengan control room untuk penyesuaian operasi.",

            # Legacy/fallback
            "Equipment": "SEGERA: Cek ASV/GCBV status. Lakukan inspeksi kompresor. Siapkan backup unit.",
            "Surge_Zone": "KRITIS: Buka Anti-Surge Valve segera! Kurangi beban kompresor. Hubungi engineer.",
            "Normal":     "Tidak ada tindakan diperlukan. Surge margin dalam batas aman.",
            "Unknown":    "Lakukan inspeksi manual dan konsultasi dengan engineer lapangan."
        }
    }
    return actions.get(equipment_type, {}).get(anomaly_class,
           "Konsultasi dengan engineer lapangan untuk evaluasi lebih lanjut.")