"""Vietnamese text normalization for WER/CER. Keeps diacritics (they change the word); drops case and punctuation."""
import re
import unicodedata

_KEEP = re.compile(r"[^0-9a-zà-ỹ\s]", re.IGNORECASE)


def normalize_vi(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = _KEEP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()
