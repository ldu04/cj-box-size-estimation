import json
from src.exceptions import SchemaValidationError

RESULT_SCHEMA = {
    "type": "object",
    "required": ["videos"],
    "properties": {
        "videos": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["video_id", "objects"],
                "properties": {
                    "video_id": {"type": "string"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["size_cm"],
                            "properties": {
                                "box_id": {"type": "integer"},
                                "size_cm": {
                                    "type": "object",
                                    "required": ["w", "d", "h"],
                                    "properties": {
                                        "w": {"type": "number", "minimum": 0},
                                        "d": {"type": "number", "minimum": 0},
                                        "h": {"type": "number", "minimum": 0},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    },
}


class ResultValidator:
    """
    Validates result.json against RESULT_SCHEMA.

    Library preference: fastjsonschema (the eval Docker env ships this,
    NOT jsonschema) → jsonschema (common dev envs) → manual structural
    check as a last resort. Missing libraries must never crash the run.
    """

    def validate(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"result.json not valid JSON: {e}") from e
        self._validate_data(data)

    def _validate_data(self, data) -> None:
        try:
            import fastjsonschema
        except ImportError:
            pass
        else:
            try:
                fastjsonschema.validate(RESULT_SCHEMA, data)
            except fastjsonschema.JsonSchemaException as e:
                raise SchemaValidationError(f"result.json invalid: {e.message}") from e
            return

        try:
            import jsonschema
        except ImportError:
            pass
        else:
            try:
                jsonschema.validate(data, RESULT_SCHEMA)
            except jsonschema.ValidationError as e:
                raise SchemaValidationError(f"result.json invalid: {e.message}") from e
            return

        self._validate_manual(data)

    @staticmethod
    def _validate_manual(data) -> None:
        if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
            raise SchemaValidationError("result.json invalid: top-level 'videos' array required")
        for v in data["videos"]:
            if not isinstance(v, dict) or not isinstance(v.get("video_id"), str) \
                    or not isinstance(v.get("objects"), list):
                raise SchemaValidationError(
                    "result.json invalid: each video needs string 'video_id' and 'objects' array"
                )
            for o in v["objects"]:
                size = o.get("size_cm") if isinstance(o, dict) else None
                if not isinstance(size, dict):
                    raise SchemaValidationError("result.json invalid: object missing 'size_cm'")
                for k in ("w", "d", "h"):
                    val = size.get(k)
                    if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                        raise SchemaValidationError(
                            f"result.json invalid: size_cm.{k} must be a non-negative number"
                        )
