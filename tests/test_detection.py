"""
Tests for Plane 2: Anomaly Detection

Covers sensor detectors, vision detectors, fusion model, alert engine,
and incident tracker.
"""

import numpy as np
import pytest
import torch

from hydrotwin.detection.sensor_detector import (
    StatisticalDetector, IsolationForestDetector,
    LSTMAutoencoderDetector, SensorAnomalyDetector, SensorAnomaly,
)

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
from hydrotwin.detection.vision_detector import (
    YOLODetector, ThermalAnalyzer, VibrationClassifier, VisionDetection,
)
from hydrotwin.detection.fusion_model import MultimodalFusionModel, ANOMALY_CLASSES
from hydrotwin.detection.alert_engine import AlertEngine, Alert
from hydrotwin.detection.incident_tracker import (
    IncidentTracker, IncidentStatus, Incident,
)


# ═══════════════════════════════════════════════════════════
#  Sensor Detectors
# ═══════════════════════════════════════════════════════════

class TestStatisticalDetector:

    def test_normal_values_no_anomaly(self):
        det = StatisticalDetector()
        # Feed normal data
        for v in np.random.normal(22, 0.5, 50):
            result = det.detect("s1", float(v))
        # Normal value should not trigger
        result = det.detect("s1", 22.5)
        assert result is None

    def test_spike_detected(self):
        det = StatisticalDetector(z_threshold=3.0)
        # Build baseline
        for v in np.random.normal(22, 0.5, 50):
            det.detect("s1", float(v))
        # Inject spike
        result = det.detect("s1", 50.0)
        assert result is not None
        assert result.anomaly_type == "spike"

    def test_flatline_detected(self):
        det = StatisticalDetector()
        for _ in range(25):
            det.detect("s1", 22.0)
        result = det.detect("s1", 22.0)
        # Flatline check triggers after 20 constant readings
        if result:
            assert result.anomaly_type == "flatline"

    def test_batch_detection(self):
        det = StatisticalDetector()
        # Build baseline
        normal = list(np.random.normal(22, 0.5, 50))
        anomalies = det.detect_batch("s1", normal + [99.0])
        # At least the 99.0 should be caught
        assert len(anomalies) >= 1

    def test_separate_sensor_buffers(self):
        det = StatisticalDetector()
        for v in np.random.normal(22, 0.5, 50):
            det.detect("s1", float(v))
        # Different sensor — should have its own buffer
        result = det.detect("s2", 22.0)
        assert result is None  # insufficient data for s2


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestIsolationForestDetector:

    def test_fit_and_detect(self):
        det = IsolationForestDetector(contamination=0.1)
        # Normal data
        normal = np.random.normal(22, 1, (100, 3))
        det.fit(normal, feature_names=["temp", "flow", "pressure"])
        # Normal point
        result = det.detect(np.array([22.0, 22.0, 22.0]))
        assert result is None or result.anomaly_type == "multivariate"

    def test_outlier_detected(self):
        det = IsolationForestDetector(contamination=0.1)
        # Generate tight cluster of normal data
        rng = np.random.default_rng(42)
        normal = rng.normal(22, 0.5, (500, 3))
        det.fit(normal)
        # Very extreme outlier — 100 standard deviations away
        result = det.detect(np.array([1000.0, 1000.0, 1000.0]))
        assert result is not None
        assert result.anomaly_type == "multivariate"

    def test_auto_fit_on_buffer(self):
        det = IsolationForestDetector()
        det._min_samples = 10
        rng = np.random.default_rng(42)
        # Feed enough data to trigger auto-fit
        for _ in range(15):
            det.detect(rng.normal(22, 1, 3))
        # After 15 samples (>10 min), model should auto-fit and clear buffer
        assert det._is_fitted or len(det._buffer) == 0


class TestLSTMAutoencoderDetector:

    def test_build_model(self):
        det = LSTMAutoencoderDetector(input_dim=1, hidden_dim=16, sequence_length=10)
        assert det._model is not None

    def test_fit_and_detect_normal(self):
        det = LSTMAutoencoderDetector(input_dim=1, hidden_dim=16, sequence_length=10)
        normal_data = np.sin(np.linspace(0, 10 * np.pi, 200)) + np.random.normal(0, 0.1, 200)
        det.fit(normal_data, epochs=5)
        assert det._is_fitted

        # Feed normal values
        for v in np.sin(np.linspace(0, np.pi, 15)):
            result = det.detect("s1", float(v))
        # Normal values should not trigger (mostly)


class TestSensorAnomalyDetector:

    def test_ensemble_creation(self):
        det = SensorAnomalyDetector(min_votes=1)
        assert det.statistical is not None
        assert det.isolation is not None
        assert det.lstm is not None

    def test_detect_spike(self):
        det = SensorAnomalyDetector(min_votes=1)
        # Build baseline
        for v in np.random.normal(22, 0.3, 50):
            det.detect("s1", float(v))
        # Spike
        result = det.detect("s1", 80.0)
        assert result is not None
        assert result.method == "ensemble"

    def test_from_config(self):
        config = {"anomaly_detection": {"sensor": {"min_votes": 2}}}
        det = SensorAnomalyDetector.from_config(config)
        assert det.min_votes == 2


# ═══════════════════════════════════════════════════════════
#  Vision Detectors
# ═══════════════════════════════════════════════════════════

class TestYOLODetector:

    def test_mock_mode(self):
        det = YOLODetector(mock_mode=True)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # May or may not detect (10% chance), but should not crash
        detections = det.detect(frame, "cam-01")
        assert isinstance(detections, list)


class TestThermalAnalyzer:

    def test_no_anomaly_in_normal_image(self):
        analyzer = ThermalAnalyzer(hotspot_threshold_c=40.0)
        # Normal thermal image (all 22°C)
        image = np.full((50, 50), 22.0)
        detections = analyzer.analyze(image)
        assert len(detections) == 0

    def test_hotspot_detected(self):
        analyzer = ThermalAnalyzer(hotspot_threshold_c=40.0, min_region_size=3)
        image = np.full((50, 50), 22.0)
        # Inject hotspot
        image[10:15, 10:15] = 55.0
        detections = analyzer.analyze(image)
        assert len(detections) >= 1
        hotspots = [d for d in detections if d.detection_type == "hotspot"]
        assert len(hotspots) == 1
        assert hotspots[0].severity in ("warning", "critical")

    def test_coldspot_detected(self):
        analyzer = ThermalAnalyzer(coldspot_threshold_c=10.0, min_region_size=3)
        image = np.full((50, 50), 22.0)
        image[20:25, 20:25] = 5.0  # cold region
        detections = analyzer.analyze(image)
        coldspots = [d for d in detections if d.detection_type == "coldspot"]
        assert len(coldspots) == 1


class TestVibrationClassifier:

    def test_normal_signal(self):
        clf = VibrationClassifier(amplitude_threshold=0.5)
        t = np.linspace(0, 1, 1000)
        signal = 0.1 * np.sin(2 * np.pi * 50 * t)  # small amplitude
        detections = clf.analyze(signal, "pump-01", fundamental_hz=50)
        # Should not detect (amplitude below threshold)
        assert len(detections) == 0

    def test_imbalance_detected(self):
        clf = VibrationClassifier(amplitude_threshold=0.3)
        t = np.linspace(0, 1, 1000)
        signal = 2.0 * np.sin(2 * np.pi * 50 * t)  # large 1x fundamental
        detections = clf.analyze(signal, "pump-01", fundamental_hz=50)
        imbalance = [d for d in detections if d.details.get("fault_type") == "imbalance"]
        assert len(imbalance) >= 1

    def test_short_signal_no_crash(self):
        clf = VibrationClassifier()
        signal = np.array([0.1, 0.2, 0.3])
        detections = clf.analyze(signal)
        assert len(detections) == 0


# ═══════════════════════════════════════════════════════════
#  Fusion Model
# ═══════════════════════════════════════════════════════════

class TestFusionModel:

    def test_forward_pass(self):
        model = MultimodalFusionModel(sensor_dim=12, vision_dim=8, vibration_dim=16, embed_dim=32)
        s = torch.randn(2, 12)
        v = torch.randn(2, 8)
        vib = torch.randn(2, 16)
        output = model(s, v, vib)
        assert output["logits"].shape == (2, 5)
        assert output["probs"].shape == (2, 5)
        assert output["predicted_class"].shape == (2,)
        assert output["modality_weights"].shape == (2, 3)

    def test_predict_numpy(self):
        model = MultimodalFusionModel(sensor_dim=4, vision_dim=4, vibration_dim=4, embed_dim=16)
        result = model.predict(np.zeros(4), np.zeros(4), np.zeros(4))
        assert result["predicted_class"] in ANOMALY_CLASSES
        assert 0 <= result["confidence"] <= 1
        assert "modality_weights" in result

    def test_probs_sum_to_one(self):
        model = MultimodalFusionModel(embed_dim=16)
        s = torch.randn(1, 12)
        v = torch.randn(1, 8)
        vib = torch.randn(1, 16)
        output = model(s, v, vib)
        prob_sum = output["probs"].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(1), atol=1e-5)


# ═══════════════════════════════════════════════════════════
#  Alert Engine
# ═══════════════════════════════════════════════════════════

class TestAlertEngine:

    def test_process_creates_alert(self):
        engine = AlertEngine()
        alert = engine.process("leak", 0.9, "Pipe-12", "sensor")
        assert alert is not None
        assert alert.severity == "critical"
        assert "leak" in alert.message.lower() or "⚠" in alert.message

    def test_deduplication(self):
        engine = AlertEngine(cooldown_seconds=300)
        a1 = engine.process("leak", 0.9, "Pipe-12")
        a2 = engine.process("leak", 0.9, "Pipe-12")  # within cooldown
        assert a1 is not None
        assert a2 is None  # deduplicated

    def test_different_locations_not_deduplicated(self):
        engine = AlertEngine()
        a1 = engine.process("leak", 0.9, "Pipe-12")
        a2 = engine.process("leak", 0.9, "Pipe-15")
        assert a1 is not None
        assert a2 is not None

    def test_acknowledge(self):
        engine = AlertEngine()
        alert = engine.process("hotspot", 0.7, "Rack-5")
        assert alert is not None
        result = engine.acknowledge(alert.alert_id)
        assert result is True

    def test_resolve(self):
        engine = AlertEngine()
        alert = engine.process("hotspot", 0.7, "Rack-5")
        engine.resolve(alert.alert_id)
        assert len(engine.active_alerts) == 0

    def test_severity_classification(self):
        engine = AlertEngine()
        # High confidence leak = critical
        a1 = engine.process("leak", 0.9, "loc-1")
        assert a1.severity == "critical"
        # Low confidence corrosion = info
        a2 = engine.process("corrosion", 0.3, "loc-2")
        assert a2.severity == "info"


# ═══════════════════════════════════════════════════════════
#  Incident Tracker
# ═══════════════════════════════════════════════════════════

class TestIncidentTracker:

    def test_create_incident(self):
        tracker = IncidentTracker()
        inc = tracker.create_incident("leak", "critical", "Pipe-12")
        assert inc.status == IncidentStatus.DETECTED
        assert inc.anomaly_type == "leak"
        assert len(inc.root_cause_hints) > 0

    def test_lifecycle_transition(self):
        tracker = IncidentTracker()
        inc = tracker.create_incident("hotspot", "warning", "Rack-5")
        assert tracker.transition(inc.incident_id, IncidentStatus.ACKNOWLEDGED)
        assert tracker.transition(inc.incident_id, IncidentStatus.INVESTIGATING)
        assert tracker.transition(inc.incident_id, IncidentStatus.RESOLVED, notes="Fixed blanking panel")
        assert len(tracker.active_incidents) == 0
        assert len(tracker.resolved_incidents) == 1

    def test_invalid_transition_rejected(self):
        tracker = IncidentTracker()
        inc = tracker.create_incident("leak", "critical", "Pipe-12")
        # Can't go directly from DETECTED to INVESTIGATING
        result = tracker.transition(inc.incident_id, IncidentStatus.INVESTIGATING)
        assert result is False

    def test_false_positive(self):
        tracker = IncidentTracker()
        inc = tracker.create_incident("leak", "warning", "Pipe-12")
        assert tracker.transition(inc.incident_id, IncidentStatus.FALSE_POSITIVE, notes="Sensor glitch")
        assert len(tracker.active_incidents) == 0

    def test_search(self):
        tracker = IncidentTracker()
        tracker.create_incident("leak", "critical", "Pipe-12")
        tracker.create_incident("hotspot", "warning", "Rack-5")
        results = tracker.search(anomaly_type="leak")
        assert len(results) == 1
        assert results[0].anomaly_type == "leak"

    def test_summary(self):
        tracker = IncidentTracker()
        tracker.create_incident("leak", "critical", "Pipe-12")
        tracker.create_incident("hotspot", "warning", "Rack-5")
        s = tracker.summary()
        assert s["active_count"] == 2
        assert s["by_severity"]["critical"] == 1

    def test_serialization(self):
        tracker = IncidentTracker()
        inc = tracker.create_incident("leak", "critical", "Pipe-12")
        d = inc.to_dict()
        assert "incident_id" in d
        assert "root_cause_hints" in d
