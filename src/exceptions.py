class CJLogisticsError(Exception):
    """Base for all project errors."""


class ModelLoadError(CJLogisticsError):
    """Raised when an ONNX model cannot be loaded."""


class CalibrationError(CJLogisticsError):
    """Raised when camera/rail calibration fails."""


class VideoReadError(CJLogisticsError):
    """Raised on unreadable or missing video."""


class SchemaValidationError(CJLogisticsError):
    """Raised when result.json fails schema validation."""


class InferenceError(CJLogisticsError):
    """Raised on ONNX runtime inference failure."""
