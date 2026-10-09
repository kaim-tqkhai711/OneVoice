"""Minimal sentence formatting for punctuation-free English ASR transcripts.

No words, numbers or internal casing are changed. This is not full punctuation
restoration; safety must still check interrogative/negation/clinical relations.
"""


def prepare_source(text, source, enabled=False):
    if not enabled or source != "en":
        return text, []
    prepared = text.strip()
    changes = []
    if prepared and "a" <= prepared[0] <= "z":
        prepared = prepared[0].upper() + prepared[1:]
        changes.append("english_initial_case")
    if prepared and prepared[-1] not in ".?!…":
        prepared += "."
        changes.append("english_terminal_period")
    return prepared, changes
