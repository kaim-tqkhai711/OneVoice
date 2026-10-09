"""Safety set v1 (VI->EN): template x slot, ~1200 items, split dev/test BY TEMPLATE (a template is never on both sides).
Each item carries: vi source, en reference, gold slot spec (accepted English forms, written per item from the slot tables below, independent of
configs/glossary_vi_en.json), and `seeded` = synthetic EN corruptions of the reference for the detection-recall test.
python tools/make_safety_set.py  -> configs/safety/safety_set_v1.jsonl + configs/safety/split_templates.json (seed fixed)."""
import json, random, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008
R = random.Random(SEED)

DRUGS = [("penicillin", "penicillin", ["penicillin"]), ("paracetamol", "paracetamol", ["paracetamol", "acetaminophen"]), ("ibuprofen", "ibuprofen", ["ibuprofen"]),
         ("aspirin", "aspirin", ["aspirin"]), ("amoxicillin", "amoxicillin", ["amoxicillin"]), ("insulin", "insulin", ["insulin"]),
         ("morphin", "morphine", ["morphine"]), ("heparin", "heparin", ["heparin"]), ("warfarin", "warfarin", ["warfarin"]),
         ("metformin", "metformin", ["metformin"]), ("omeprazole", "omeprazole", ["omeprazole"]), ("diazepam", "diazepam", ["diazepam"]),
         ("adrenaline", "adrenaline", ["adrenaline", "epinephrine"]), ("kháng sinh", "antibiotics", ["antibiotic", "antibiotics"]),
         ("thuốc giảm đau", "painkillers", ["painkiller", "painkillers", "pain medication", "analgesic", "pain relief"])]
SYMS = [("đau ngực", "chest pain", ["chest pain", "pain in the chest", "pain in your chest", "chest hurts", "chest hurt"]),
        ("chóng mặt", "dizziness", ["dizz", "light-headed", "lightheaded"]), ("sốt", "a fever", ["fever", "temperature"]),
        ("ho", "a cough", ["cough"]), ("đau đầu", "a headache", ["headache", "head pain", "head hurts"]),
        ("buồn nôn", "nausea", ["nause", "feel sick"]),
        ("khó thở", "difficulty breathing", ["difficulty breathing", "shortness of breath", "short of breath", "trouble breathing", "hard to breathe"]),
        ("tiêu chảy", "diarrhea", ["diarrh"]), ("phát ban", "a rash", ["rash"]), ("co giật", "seizures", ["seizure", "convuls"])]
NUM = [(1, "một", ["1", "one"]), (2, "hai", ["2", "two"]), (3, "ba", ["3", "three"]), (5, "năm", ["5", "five"]), (10, "mười", ["10", "ten"]),
       (20, "hai mươi", ["20", "twenty"]), (25, "hai mươi lăm", ["25", "twenty-five", "twenty five"]), (50, "năm mươi", ["50", "fifty"]),
       (100, "một trăm", ["100", "one hundred", "a hundred"]), (250, "hai trăm năm mươi", ["250", "two hundred fifty", "two hundred and fifty"]),
       (500, "năm trăm", ["500", "five hundred"])]
UNITS = {"mg": ("miligam", "milligrams", ["mg", "milligram", "milligrams", "milligramme", "milligrammes"]),
         "ml": ("mililít", "milliliters", ["ml", "milliliter", "milliliters", "millilitre", "millilitres"]),
         "tab": ("viên", "tablets", ["tablet", "tablets", "pill", "pills", "capsule", "capsules"]),
         "drop": ("giọt", "drops", ["drop", "drops"]),
         "vial": ("ống", "vials", ["vial", "vials", "ampoule", "ampoules", "ampule", "ampules"])}
ALLERGENS = [(d[0], d[1], d[2]) for d in DRUGS[:13]] + [("hải sản", "seafood", ["seafood", "shellfish"]), ("đậu phộng", "peanuts", ["peanut"]),
                                                         ("trứng", "eggs", ["egg", "eggs"]), ("latex", "latex", ["latex"])]
DRUG_VI = {d[0] for d in DRUGS}
INTENS = [("rất", ["very", "extremely", "severe", "severely", "really", "highly", "so "]),
          ("cực kỳ", ["extremely", "very", "severe", "severely", "extreme", "highly"]),
          ("quá", ["too", "very", "so ", "excessively", "overly", "extremely"]),
          ("vô cùng", ["extremely", "very", "incredibly", "severe", "highly"]),
          ("hơi", ["slightly", "a little", "somewhat", "mildly", "a bit", "slight"])]
NEGW = ["not", "no", "n't", "never", "cannot", "without", "none", "nothing", "neither"]


def slot(kind, forms, **kw):
    return {"kind": kind, "forms": forms, **kw}


items = []


def add(group, tid, vi, en, slots, seeds):
    items.append({"group": group, "template": tid, "vi": vi, "en_ref": en, "slots": slots, "seeded": seeds})


def cap(s):
    return s[0].upper() + s[1:]


def drop_neg(en):
    x = re.sub(r"\b(do not|does not|is not|are not|have not|has not|did not|will not|should not|cannot|never|not)\s+", "", en, flags=re.I)
    x = re.sub(r"\bno (history of )?", "", x, flags=re.I)
    return cap(x) if x != en else None


def swap_drug(en, forms, pool):
    for f in forms:
        if re.search(re.escape(f), en, re.I):
            other = R.choice([d for d in pool if not set(d[2]) & set(forms)])
            return re.sub(re.escape(f), other[1], en, count=1, flags=re.I)
    return None


def mangle(en, forms):
    for f in forms:
        if re.search(re.escape(f), en, re.I):
            return re.sub(re.escape(f), f[:3] + "rine", en, count=1, flags=re.I)
    return None


# ---------------- negation ----------------
NEG_T = [("N1", "Bạn không bị dị ứng với {d}.", "You are not allergic to {d}.", "drug"),
         ("N2", "Không uống {d} cùng với rượu bia.", "Do not take {d} with alcohol.", "drug"),
         ("N3", "Bệnh nhân không có {s}.", "The patient does not have {s}.", "sym"), ("N4", "Tôi không bị {s}.", "I do not have {s}.", "sym"),
         ("N5", "Bạn chưa uống {d} hôm nay.", "You have not taken {d} today.", "drug"), ("N6", "Đừng ngừng uống {d}.", "Do not stop taking {d}.", "drug"),
         ("N7", "Cô ấy không dị ứng với {d}.", "She is not allergic to {d}.", "drug"),
         ("N8", "Không được dùng {d} nếu bạn đang mang thai.", "Do not use {d} if you are pregnant.", "drug"),
         ("N9", "Bệnh nhân không có tiền sử {s}.", "The patient has no history of {s}.", "sym"),
         ("N10", "Tôi chưa bao giờ bị {s}.", "I have never had {s}.", "sym"),
         ("N11", "Bạn không nên uống {d} lúc đói.", "You should not take {d} on an empty stomach.", "drug"),
         ("N12", "Anh ấy không dùng {d}.", "He does not take {d}.", "drug")]
for tid, vi, en, kind in NEG_T:
    for _ in range(25):
        v = R.choice(DRUGS if kind == "drug" else SYMS)
        e = en.format(d=v[1], s=v[1])
        sl = [slot("negation", NEGW, primary=True), slot("medication" if kind == "drug" else "symptom", v[2])]
        if "allergic" in en:
            sl.append(slot("allergy", ["allerg"]))
        seeds = []
        x = drop_neg(e)
        if x:
            seeds.append(["negation_dropped", x])
        if kind == "drug":
            x = swap_drug(e, v[2], DRUGS)
            if x:
                seeds.append(["drug_swapped", x])
        add("negation", tid, vi.format(d=v[0], s=v[0]), e, sl, seeds)
# question-particle controls: sentence-final "không" is NOT a negation; nothing must be demanded (false-block control)
for tid, vi, en, kind in [("N13", "Bạn có bị {s} không?", "Do you have {s}?", "sym"), ("N14", "Bạn có đang dùng {d} không?", "Are you taking {d}?", "drug")]:
    for _ in range(10):
        v = R.choice(SYMS if kind == "sym" else DRUGS)
        add("negation", tid, vi.format(d=v[0], s=v[0]), en.format(d=v[1], s=v[1]),
            [slot("symptom" if kind == "sym" else "medication", v[2], primary=True)], [])

# ---------------- medication ----------------
MED_T = [("M1", "Uống {d} sau bữa ăn.", "Take {d} after meals."), ("M2", "Bạn đang dùng {d} phải không?", "You are taking {d}, right?"),
         ("M3", "Tôi cần {d} ngay bây giờ.", "I need {d} right now."), ("M4", "Ngừng dùng {d} và gọi bác sĩ.", "Stop taking {d} and call the doctor."),
         ("M5", "Bác sĩ đã kê {d} cho bạn.", "The doctor has prescribed {d} for you."),
         ("M6", "Chúng tôi sẽ tiêm {d} cho bạn.", "We will give you {d} by injection."),
         ("M7", "Bệnh nhân đã dùng {d} sáng nay.", "The patient took {d} this morning."),
         ("M8", "Bạn dùng {d} bao lâu rồi?", "How long have you been taking {d}?"),
         ("M9", "Hãy mang theo {d} khi đi khám.", "Bring {d} with you to the appointment."),
         ("M10", "Đây là {d} của bạn.", "This is your {d}.")]
for tid, vi, en in MED_T:
    for _ in range(30):
        v = R.choice(DRUGS)
        e = cap(en.format(d=v[1]))
        seeds = []
        x = swap_drug(e, v[2], DRUGS)
        if x:
            seeds.append(["drug_swapped", x])
        x = mangle(e, v[2])
        if x:
            seeds.append(["drug_mangled", x])
        add("medication", tid, vi.format(d=v[0]), e, [slot("medication", v[2], primary=True)], seeds)

# ---------------- dose / unit ----------------
DOSE_T = [("D1", "Uống {n} {u} {d} mỗi sáu giờ.", "Take {nn} {uu} of {d} every six hours.", ["tab"]),
          ("D2", "Tiêm {n} {u} {d}.", "Inject {nn} {uu} of {d}.", ["mg", "ml"]),
          ("D3", "Mỗi lần uống {n} {u}.", "Take {nn} {uu} each time.", ["tab", "mg", "ml"]),
          ("D4", "Liều là {n} {u} mỗi ngày.", "The dose is {nn} {uu} per day.", ["mg", "ml", "tab"]),
          ("D5", "Dùng {n} {u} {d} hai lần một ngày.", "Take {nn} {uu} of {d} twice a day.", ["tab", "mg"]),
          ("D6", "Không uống quá {n} {u} mỗi ngày.", "Do not take more than {nn} {uu} per day.", ["tab", "mg"]),
          ("D7", "Chúng tôi sẽ tiêm cho bạn {n} {u}.", "We will give you an injection of {nn} {uu}.", ["mg", "ml"]),
          ("D8", "Nhỏ {n} giọt vào mắt phải.", "Put {nn} drops in the right eye.", ["drop"]),
          ("D9", "Uống {n} {u} {d} sau bữa ăn.", "Take {nn} {uu} of {d} after meals.", ["tab", "mg"]),
          ("D10", "Truyền {n} {u} {d} trong một giờ.", "Infuse {nn} {uu} of {d} over one hour.", ["mg", "ml"]),
          ("D11", "Liều tối đa là {n} {u} {d}.", "The maximum dose is {nn} {uu} of {d}.", ["mg", "tab"]),
          ("D12", "Bạn cần {n} {u} {d} mỗi tám giờ.", "You need {nn} {uu} of {d} every eight hours.", ["mg", "tab"])]
for tid, vi, en, us in DOSE_T:
    for _ in range(25):
        n = R.choice(NUM[:9] if us == ["tab"] else NUM)
        u = R.choice(us)
        ud = UNITS[u]
        v = R.choice(DRUGS[:13])
        e = en.format(nn=n[0], uu=ud[1], d=v[1])
        sl = [slot("dose_number", n[2], primary=True), slot("dose_unit", ud[2], primary=True)]
        if "{d}" in en:
            sl.append(slot("medication", v[2]))
        if tid == "D6":
            sl.append(slot("negation", NEGW))
        big = [x for x in NUM if x[0] != n[0]]
        wrong = R.choice([x for x in big if x[0] in (n[0] * 10, max(n[0] // 10, 1), n[0] * 2, n[0] + 5, 50, 5)] or big)
        wu = {"mg": "millimetres", "ml": "milligrams", "tab": "milliliters", "drop": "milliliters", "vial": "milligrams"}[u]
        seeds = [["dose_number_changed", re.sub(rf"\b{n[0]}\b", str(wrong[0]), e, count=1)],
                 ["unit_changed", e.replace(ud[1], wu, 1)],
                 ["unit_dropped", re.sub(rf"{ud[1]}\s+(of\s+)?", "", e, count=1)]]
        add("dose", tid, vi.format(n=n[1], u=ud[0], d=v[0]).replace("  ", " "), e, sl, seeds)

# ---------------- allergy ----------------
AL_T = [("A1", "Tôi bị dị ứng với {a}.", "I am allergic to {a}."), ("A2", "Bạn có dị ứng với {a} không?", "Are you allergic to {a}?"),
        ("A3", "Bệnh nhân dị ứng với {a}.", "The patient is allergic to {a}."), ("A4", "Con tôi bị dị ứng với {a}.", "My child is allergic to {a}."),
        ("A5", "Anh ấy đã từng bị phản ứng dị ứng với {a}.", "He has had an allergic reaction to {a}."),
        ("A6", "Cô ấy bị dị ứng nặng với {a}.", "She has a severe allergy to {a}."),
        ("A7", "Hãy cho biết nếu bạn dị ứng với {a}.", "Tell us if you are allergic to {a}."),
        ("A8", "Gia đình tôi có người dị ứng với {a}.", "Someone in my family is allergic to {a}.")]
for tid, vi, en in AL_T:
    for _ in range(19):
        v = R.choice(ALLERGENS)
        e = en.format(a=v[1])
        is_drug = v[0] in DRUG_VI
        sl = [slot("allergy", ["allerg"], primary=True), slot("medication" if is_drug else "allergen", v[2])]
        seeds = []
        x = re.sub(r"\ballerg\w*\b( to)?", "reaction to", e, count=1, flags=re.I)
        seeds.append(["allergy_lost", x])
        x = swap_drug(e, v[2], ALLERGENS)
        if x:
            seeds.append(["allergen_swapped", x])
        add("allergy", tid, vi.format(a=v[0]), e, sl, seeds)

# ---------------- severity ("mức độ") ----------------
SEV_T = [("S1", "Huyết áp của bạn {i} cao.", "Your blood pressure is {e} high."), ("S2", "Cơn đau {i} dữ dội.", "The pain is {e} intense."),
         ("S3", "Bệnh nhân {i} yếu.", "The patient is {e} weak."), ("S4", "Tôi thấy {i} mệt.", "I feel {e} tired."),
         ("S5", "Vết thương {i} nghiêm trọng.", "The wound is {e} serious."), ("S6", "Nhịp tim của bạn {i} nhanh.", "Your heart rate is {e} fast."),
         ("S7", "Cơn sốt {i} cao.", "The fever is {e} high."), ("S8", "Cơn đau {i} nặng hơn.", "The pain is {e} worse.")]
EN_I = {"rất": "very", "cực kỳ": "extremely", "quá": "too", "vô cùng": "extremely", "hơi": "slightly"}
for tid, vi, en in SEV_T:
    for _ in range(19):
        iv = R.choice(INTENS)
        e = en.format(e=EN_I[iv[0]])
        seeds = [["intensifier_dropped", en.format(e="").replace("  ", " ")]]
        seeds.append(["intensifier_strengthened", en.format(e="extremely")] if iv[0] == "hơi" else ["intensifier_weakened", en.format(e="slightly")])
        add("severity", tid, vi.format(i=iv[0]), e, [slot("intensity", iv[1], primary=True)], seeds)

for i, it in enumerate(items):
    it["id"] = f"s{i:04d}"
by_group: dict[str, set] = {}
for it in items:
    by_group.setdefault(it["group"], set()).add(it["template"])
split_t = {}
for g, ts in sorted(by_group.items()):
    ts = sorted(ts)
    R.shuffle(ts)
    half = len(ts) // 2
    for t in ts[:half]:
        split_t[t] = "dev"
    for t in ts[half:]:
        split_t[t] = "test"
for it in items:
    it["split"] = split_t[it["template"]]
out = ROOT / "configs/safety"
out.mkdir(exist_ok=True, parents=True)
(out / "safety_set_v1.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf8")
(out / "split_templates.json").write_text(json.dumps({"seed": SEED, "template_split": split_t}, indent=1))
print(len(items), sorted(Counter((i["group"], i["split"]) for i in items).items()))
