from tonebridge.nmt_constraints import GlossaryConstrainer


def test_dose_limit_not_intensity():
    forms = GlossaryConstrainer().forms("không uống quá hai viên")
    assert ["too", "very"] not in forms
    assert ["2"] in forms


def test_decimal_constraint_not_integer_part():
    forms = GlossaryConstrainer().forms("tiêm 0.5 miligam morphin")
    assert ["0.5"] in forms and ["5"] not in forms


def test_prednisone_not_prednisolone_alias():
    forms = GlossaryConstrainer().forms("uống prednisolone")
    assert ["prednisolone"] in forms
    assert not any("prednisone" in alternatives for alternatives in forms)
