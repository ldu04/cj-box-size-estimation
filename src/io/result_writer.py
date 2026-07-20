from src.schema import CompetitionResult


class ResultWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def write(self, result: CompetitionResult) -> None:
        result.to_json(self.output_path)
