from typing import List, TypedDict


class InputJSON(TypedDict):
    text: str
    text_hash: str
    text_source: str

class Paraphrase(TypedDict):
    paraphrase: str

class OutputJSON(InputJSON):
    paraphrases: List[Paraphrase]