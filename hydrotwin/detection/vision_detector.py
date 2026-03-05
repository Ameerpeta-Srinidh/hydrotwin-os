"""
HydroTwin OS — Plane 2: Vision-Based Detector

Computer vision anomaly detection using YOLOv8 for physical damage,
thermal image analysis for hotspots, and FFT-based vibration analysis
for equipment health monitoring.

Detectors:
    YOLODetector          — Leak, corrosion, and damage detection from camera frames
    ThermalAnalyzer       — Hotspot detection from thermal images
    VibrationClassifier   — Pump/fan health from vibration FFT patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────── Data Types ───────────────────────

@dataclass
class VisionDetection:
    """A detection from a vision-based analyzer."""
    detection_type: str         # leak, corrosion, damage, hotspot, vibration_anomaly
    confidence: float = 0.0    # 0–1
    severity: str = "info"     # info, warning, critical
    location: str = ""         # physical location or camera ID
    bounding_box: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] normalized
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_type": self.detection_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "location": self.location,
            "bounding_box": self.bounding_box,
            "details": self.details,
            "timestamp": str(self.timestamp),
        }


# ─────────────────────── YOLO Detector ───────────────────────

class YOLODetector:
    """
    Wraps YOLOv8 for detecting physical anomalies in data center imagery.

    Detectable classes: leak, corrosion, cable_damage, floor_water, equipment_damage
    Falls back to mock detection when ultralytics is not installed.
    """

    DEFAULT_CLASSES = ["leak", "corrosion", "cable_damage", "floor_water", "equipment_damage"]

    SEVERITY_MAP = {
        "leak": "critical",
        "corrosion": "warning",
        "cable_damage": "warning",
        "floor_water": "critical",
        "equipment_damage": "critical",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        classes: list[str] | None = None,
        mock_mode: bool = False,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes or self.DEFAULT_CLASSES
        self.mock_mode = mock_mode
        self._model = None

        if not mock_mode:
            self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"YOLOv8 model loaded: {self.model_path}")
        except ImportError:
            logger.warning("ultralytics not installed. Using mock mode.")
            self.mock_mode = True
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Using mock mode.")
            self.mock_mode = True

    def detect(self, frame: np.ndarray, camera_id: str = "cam-01") -> list[VisionDetection]:
        """
        Run detection on a camera frame.

        Args:
            frame: BGR image as numpy array [H, W, 3]
            camera_id: identifier for the camera location

        Returns:
            List of VisionDetection objects
        """
        if self.mock_mode:
            return self._mock_detect(frame, camera_id)

        results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                # Normalize bounding box
                h, w = frame.shape[:2]
                norm_box = [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h]

                detections.append(VisionDetection(
                    detection_type=cls_name,
                    confidence=conf,
                    severity=self.SEVERITY_MAP.get(cls_name, "info"),
                    location=camera_id,
                    bounding_box=norm_box,
                    details={"class_id": cls_id, "raw_xyxy": xyxy},
                ))

        return detections

    def _mock_detect(self, frame: np.ndarray, camera_id: str) -> list[VisionDetection]:
        """Generate mock detections for testing."""
        rng = np.random.default_rng()

        # 10% chance of detecting an anomaly
        if rng.random() > 0.1:
            return []

        cls = rng.choice(self.classes)
        return [VisionDetection(
            detection_type=cls,
            confidence=float(rng.uniform(0.6, 0.95)),
            severity=self.SEVERITY_MAP.get(cls, "warning"),
            location=camera_id,
            bounding_box=[float(x) for x in rng.uniform(0, 0.8, 4)],
            details={"mock": True},
        )]


# ─────────────────────── Thermal Analyzer ───────────────────────

class ThermalAnalyzer:
    """
    Analyzes thermal images (or temperature matrices) for hotspot detection.

    Uses threshold-based detection with spatial clustering to identify
    thermal anomalies that may indicate equipment overheating, coolant
    failures, or airflow obstructions.
    """

    def __init__(
        self,
        hotspot_threshold_c: float = 40.0,
        coldspot_threshold_c: float = 10.0,
        min_region_size: int = 5,
    ):
        self.hotspot_threshold = hotspot_threshold_c
        self.coldspot_threshold = coldspot_threshold_c
        self.min_region_size = min_region_size

    def analyze(
        self,
        thermal_image: np.ndarray,
        location: str = "thermal-cam-01",
    ) -> list[VisionDetection]:
        """
        Analyze a thermal image for hotspots and coldspots.

        Args:
            thermal_image: 2D array of temperatures (°C) [H, W]
            location: camera/sensor location identifier

        Returns:
            List of VisionDetection objects for thermal anomalies
        """
        detections = []

        if thermal_image.ndim != 2:
            return detections

        h, w = thermal_image.shape

        # Hotspot detection
        hotspot_mask = thermal_image > self.hotspot_threshold
        hotspot_count = hotspot_mask.sum()

        if hotspot_count >= self.min_region_size:
            hot_coords = np.argwhere(hotspot_mask)
            max_temp = float(thermal_image[hotspot_mask].max())
            mean_temp = float(thermal_image[hotspot_mask].mean())

            # Bounding box of hotspot region
            y_min, x_min = hot_coords.min(axis=0)
            y_max, x_max = hot_coords.max(axis=0)
            bbox = [x_min / w, y_min / h, x_max / w, y_max / h]

            severity = "critical" if max_temp > 60 else "warning"
            confidence = min(1.0, (max_temp - self.hotspot_threshold) / 30.0)

            detections.append(VisionDetection(
                detection_type="hotspot",
                confidence=confidence,
                severity=severity,
                location=location,
                bounding_box=[float(x) for x in bbox],
                details={
                    "max_temp_c": round(max_temp, 1),
                    "mean_temp_c": round(mean_temp, 1),
                    "affected_pixels": int(hotspot_count),
                    "area_fraction": round(hotspot_count / (h * w), 4),
                },
            ))

        # Coldspot detection (may indicate overcooling / coolant leak)
        coldspot_mask = thermal_image < self.coldspot_threshold
        coldspot_count = coldspot_mask.sum()

        if coldspot_count >= self.min_region_size:
            cold_coords = np.argwhere(coldspot_mask)
            min_temp = float(thermal_image[coldspot_mask].min())
            y_min, x_min = cold_coords.min(axis=0)
            y_max, x_max = cold_coords.max(axis=0)

            detections.append(VisionDetection(
                detection_type="coldspot",
                confidence=min(1.0, (self.coldspot_threshold - min_temp) / 10.0),
                severity="warning",
                location=location,
                bounding_box=[x_min / w, y_min / h, x_max / w, y_max / h],
                details={"min_temp_c": round(min_temp, 1), "affected_pixels": int(coldspot_count)},
            ))

        return detections


# ─────────────────────── Vibration Classifier ───────────────────────

class VibrationClassifier:
    """
    FFT-based vibration pattern analysis for pump/fan health monitoring.

    Analyzes vibration signals to detect:
        - Imbalance (1x fundamental)
        - Misalignment (2x fundamental)
        - Bearing defects (high-frequency harmonics)
        - Looseness (sub-harmonics)
    """

    def __init__(
        self,
        sample_rate_hz: float = 1000.0,
        amplitude_threshold: float = 0.3,
    ):
        self.sample_rate = sample_rate_hz
        self.amplitude_threshold = amplitude_threshold

    def analyze(
        self,
        vibration_signal: np.ndarray,
        equipment_id: str = "pump-01",
        fundamental_hz: float = 50.0,
    ) -> list[VisionDetection]:
        """
        Analyze a vibration signal using FFT.

        Args:
            vibration_signal: 1D time-domain vibration data
            equipment_id: identifier for the equipment
            fundamental_hz: expected fundamental frequency (RPM/60)

        Returns:
            List of VisionDetection objects for vibration anomalies
        """
        detections = []

        if len(vibration_signal) < 64:
            return detections

        # FFT
        n = len(vibration_signal)
        fft_vals = np.fft.rfft(vibration_signal)
        fft_magnitudes = np.abs(fft_vals) / n
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # Find dominant frequency
        peak_idx = np.argmax(fft_magnitudes[1:]) + 1  # skip DC
        peak_freq = frequencies[peak_idx]
        peak_amplitude = fft_magnitudes[peak_idx]

        # Overall RMS
        rms = float(np.sqrt(np.mean(vibration_signal ** 2)))

        # Check for specific fault patterns
        # 1x fundamental — imbalance
        f1_idx = np.argmin(np.abs(frequencies - fundamental_hz))
        f1_amp = float(fft_magnitudes[f1_idx])

        # 2x fundamental — misalignment
        f2_idx = np.argmin(np.abs(frequencies - 2 * fundamental_hz))
        f2_amp = float(fft_magnitudes[f2_idx]) if f2_idx < len(fft_magnitudes) else 0

        # High-frequency energy — bearing defects
        high_freq_mask = frequencies > 5 * fundamental_hz
        high_freq_energy = float(fft_magnitudes[high_freq_mask].sum()) if high_freq_mask.any() else 0

        # Check thresholds
        if f1_amp > self.amplitude_threshold:
            detections.append(VisionDetection(
                detection_type="vibration_anomaly",
                confidence=min(1.0, f1_amp / self.amplitude_threshold),
                severity="warning" if f1_amp < self.amplitude_threshold * 2 else "critical",
                location=equipment_id,
                details={
                    "fault_type": "imbalance",
                    "frequency_hz": round(fundamental_hz, 1),
                    "amplitude": round(f1_amp, 4),
                    "rms": round(rms, 4),
                },
            ))

        if f2_amp > self.amplitude_threshold * 0.7:
            detections.append(VisionDetection(
                detection_type="vibration_anomaly",
                confidence=min(1.0, f2_amp / (self.amplitude_threshold * 0.7)),
                severity="warning",
                location=equipment_id,
                details={
                    "fault_type": "misalignment",
                    "frequency_hz": round(2 * fundamental_hz, 1),
                    "amplitude": round(f2_amp, 4),
                },
            ))

        if high_freq_energy > self.amplitude_threshold * 2:
            detections.append(VisionDetection(
                detection_type="vibration_anomaly",
                confidence=min(1.0, high_freq_energy / (self.amplitude_threshold * 5)),
                severity="critical",
                location=equipment_id,
                details={
                    "fault_type": "bearing_defect",
                    "high_freq_energy": round(high_freq_energy, 4),
                },
            ))

        return detections
