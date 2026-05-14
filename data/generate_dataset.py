"""
Healthcare Claims Dataset Generator
====================================
Generates synthetic 2024 claims data with five embedded fraud patterns.

Fraud patterns:
    1. GHOST_BILLING     — PRV-047 bills for 4 deceased patients
    2. REFERRAL_RING     — PRV-031/032/033 form a closed referral loop with inflated claims
    3. IMPOSSIBLE_TRAVEL — PAT-0089 has same-day claims in cities 1,300 miles apart
    4. UPCODING          — PRV-022 bills high-complexity code 99215 at 8× the normal rate
    5. DUPLICATE_BILLING — PRV-055 re-submits the same claim 1–2 extra times

Output files:
    data/claims.csv         — full dataset WITH is_fraud / fraud_type labels  (instructor)
    data/claims_public.csv  — same rows, labels stripped                       (participants)
    data/patients.csv
    data/providers.csv
    data/referrals.csv

Usage:
    python data/generate_dataset.py
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
fake = Faker()
Faker.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_PATIENTS  = 300
N_PROVIDERS = 60
DATE_START  = datetime(2024, 1, 1)
DATE_END    = datetime(2024, 12, 31)

# (code, description, base_usd, typical_frequency)
PROCEDURE_CODES = [
    ("99213", "Office visit, established, low complexity",      150, 0.35),
    ("99214", "Office visit, established, moderate complexity", 220, 0.25),
    ("99215", "Office visit, established, high complexity",     320, 0.12),
    ("99203", "Office visit, new patient, low complexity",      180, 0.10),
    ("99204", "Office visit, new patient, moderate complexity", 260, 0.08),
    ("80053", "Comprehensive metabolic panel",                   45, 0.04),
    ("85025", "Complete blood count",                            35, 0.03),
    ("93000", "Electrocardiogram",                               75, 0.02),
    ("99283", "ED visit, moderate severity",                    380, 0.01),
]
PROC_WEIGHTS = [p[3] for p in PROCEDURE_CODES]

DIAGNOSIS_CODES = [
    "I10", "E11.9", "J06.9", "M54.5", "F32.9",
    "Z00.00", "K21.0", "N39.0", "J45.909", "M79.3",
]

SPECIALTIES = [
    "Internal Medicine", "Family Medicine", "Cardiology",
    "Orthopedics", "Neurology", "Gastroenterology",
    "Dermatology", "Endocrinology", "Pulmonology", "Rheumatology",
]

CITY_BY_STATE = {
    "NY": "New York",    "CA": "Los Angeles", "IL": "Chicago",
    "TX": "Houston",     "FL": "Miami",       "WA": "Seattle",
    "MA": "Boston",      "GA": "Atlanta",     "CO": "Denver",
    "AZ": "Phoenix",
}
STATES = list(CITY_BY_STATE.keys())

# ── ID formatters ──────────────────────────────────────────────────────────────
def pat_id(n):  return f"PAT-{n:04d}"
def prv_id(n):  return f"PRV-{n:03d}"
def clm_id(n):  return f"CLM-{n:06d}"
def ref_id(n):  return f"REF-{n:05d}"

def rand_date(start: datetime = DATE_START, end: datetime = DATE_END) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))

# ── Patients ───────────────────────────────────────────────────────────────────
def generate_patients() -> pd.DataFrame:
    rows = []
    for i in range(1, N_PATIENTS + 1):
        state = random.choice(STATES)
        dob   = rand_date(datetime(1940, 1, 1), datetime(1995, 12, 31))
        # Patients 295-298 died in 2023 — ghost-billing targets
        dod = (rand_date(datetime(2023, 1, 1), datetime(2023, 12, 31))
               if i in (295, 296, 297, 298) else None)
        rows.append({
            "patient_id":    pat_id(i),
            "first_name":    fake.first_name(),
            "last_name":     fake.last_name(),
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "gender":        random.choice(["M", "F"]),
            "city":          CITY_BY_STATE[state],
            "state":         state,
            "zip_code":      fake.zipcode(),
            "insurance_id":  f"INS{random.randint(100_000, 999_999)}",
            "date_of_death": dod.strftime("%Y-%m-%d") if dod else None,
        })
    return pd.DataFrame(rows)

# ── Providers ──────────────────────────────────────────────────────────────────
def generate_providers() -> pd.DataFrame:
    FRAUD_FIXED = {
        22: ("Dr. Linda Chen",           "Family Medicine",   "NY", "New York"),
        31: ("Sunrise Medical Group",    "Internal Medicine", "FL", "Miami"),
        32: ("Valley Diagnostics LLC",   "Gastroenterology",  "FL", "Miami"),
        33: ("Metro Health Specialists", "Cardiology",        "FL", "Miami"),
        47: ("Dr. Richard Morrow",       "Internal Medicine", "CA", "Los Angeles"),
        55: ("Coastal Medical Group",    "Family Medicine",   "TX", "Houston"),
    }
    rows = []
    for i in range(1, N_PROVIDERS + 1):
        if i in FRAUD_FIXED:
            name, spec, state, city = FRAUD_FIXED[i]
            is_group = any(w in name for w in ("Group", "LLC", "Medical", "Diagnostics", "Specialists"))
            ptype = "Group" if is_group else "Individual"
        else:
            state = random.choice(STATES)
            city  = CITY_BY_STATE[state]
            spec  = random.choice(SPECIALTIES)
            if random.random() < 0.30:
                name  = f"{fake.company()} {random.choice(['Medical', 'Health', 'Clinic'])}"
                ptype = "Group"
            else:
                name  = f"Dr. {fake.first_name()} {fake.last_name()}"
                ptype = "Individual"
        rows.append({
            "provider_id":   prv_id(i),
            "provider_name": name,
            "specialty":     spec,
            "city":          city,
            "state":         state,
            "npi_number":    "".join(str(random.randint(0, 9)) for _ in range(10)),
            "provider_type": ptype,
        })
    return pd.DataFrame(rows)

# ── Claim row factory ──────────────────────────────────────────────────────────
def make_claim(ctr: int, patient_id: str, provider_id: str, svc_date: datetime,
               is_fraud: bool, fraud_type, amount_mult: float = 1.0,
               proc_override=None) -> dict:
    proc = proc_override or random.choices(PROCEDURE_CODES, weights=PROC_WEIGHTS, k=1)[0]
    amt  = proc[2] * random.uniform(0.9, 1.3) * amount_mult
    return {
        "claim_id":        clm_id(ctr),
        "patient_id":      patient_id,
        "provider_id":     provider_id,
        "procedure_code":  proc[0],
        "procedure_desc":  proc[1],
        "diagnosis_code":  random.choice(DIAGNOSIS_CODES),
        "date_of_service": svc_date.strftime("%Y-%m-%d"),
        "amount_billed":   round(amt, 2),
        "amount_paid":     round(amt * 0.80, 2),
        "claim_status":    "Approved",
        "is_fraud":        is_fraud,
        "fraud_type":      fraud_type,
    }

# ── Normal claims ──────────────────────────────────────────────────────────────
def generate_normal_claims(patients: pd.DataFrame, providers: pd.DataFrame, ctr: int):
    FRAUD_PROVIDERS = {"PRV-022", "PRV-031", "PRV-032", "PRV-033", "PRV-047", "PRV-055"}
    GHOST_PATIENTS  = {pat_id(i) for i in (295, 296, 297, 298)}
    pool_prov = providers[~providers["provider_id"].isin(FRAUD_PROVIDERS)]
    pool_pat  = patients[~patients["patient_id"].isin(GHOST_PATIENTS)]
    rows = []
    for _ in range(1_800):
        pat  = pool_pat.sample(1).iloc[0]
        prov = pool_prov.sample(1).iloc[0]
        rows.append(make_claim(ctr, pat["patient_id"], prov["provider_id"],
                               rand_date(), False, None))
        ctr += 1
    return rows, ctr

# ── Fraud 1: Ghost Billing ─────────────────────────────────────────────────────
def generate_ghost_billing(patients: pd.DataFrame, ctr: int):
    """PRV-047 files 4–6 claims per deceased patient, all dated after their death."""
    ghost_pats = patients[patients["patient_id"].isin({pat_id(i) for i in (295, 296, 297, 298)})]
    rows = []
    for _, pat in ghost_pats.iterrows():
        for _ in range(random.randint(4, 6)):
            svc = rand_date(datetime(2024, 3, 1), DATE_END)
            rows.append(make_claim(ctr, pat["patient_id"], "PRV-047",
                                   svc, True, "GHOST_BILLING", 1.2))
            ctr += 1
    return rows, ctr

# ── Fraud 2: Referral Ring ─────────────────────────────────────────────────────
def generate_referral_ring(patients: pd.DataFrame, ctr: int):
    """PRV-031/032/033 each bill the same 15 patients at 2–3× normal rates."""
    ring_pats = patients[patients["patient_id"].isin({pat_id(i) for i in range(50, 65)})]
    rows = []
    for _, pat in ring_pats.iterrows():
        for prov in ("PRV-031", "PRV-032", "PRV-033"):
            for _ in range(random.randint(3, 6)):
                rows.append(make_claim(ctr, pat["patient_id"], prov, rand_date(),
                                       True, "REFERRAL_RING",
                                       random.uniform(2.0, 3.5)))
                ctr += 1
    return rows, ctr

# ── Fraud 3: Impossible Travel ─────────────────────────────────────────────────
def generate_impossible_travel(patients: pd.DataFrame, ctr: int):
    """PAT-0089 has same-day claims at PRV-007 (New York) and PRV-047 (Los Angeles)."""
    travel_dates = [
        datetime(2024, 2, 14), datetime(2024, 4,  3), datetime(2024, 6, 19),
        datetime(2024, 8,  8), datetime(2024, 10, 22),
    ]
    rows = []
    for dt in travel_dates:
        for prov in ("PRV-007", "PRV-047"):
            rows.append(make_claim(ctr, "PAT-0089", prov, dt, True, "IMPOSSIBLE_TRAVEL"))
            ctr += 1
    return rows, ctr

# ── Fraud 4: Upcoding ─────────────────────────────────────────────────────────
def generate_upcoding(patients: pd.DataFrame, ctr: int):
    """PRV-022 bills 99215 (highest complexity, $320) for ~88 % of visits; normal ≈ 12 %."""
    pool = patients[patients["patient_id"].isin({pat_id(i) for i in range(100, 140)})]
    rows = []
    for _ in range(120):
        pat = pool.sample(1).iloc[0]
        if random.random() < 0.88:
            proc = ("99215", "Office visit, established, high complexity", 320, 0.12)
        else:
            proc = random.choices(PROCEDURE_CODES[:4], k=1)[0]
        amt = proc[2] * random.uniform(1.0, 1.2)
        rows.append({
            "claim_id":        clm_id(ctr),
            "patient_id":      pat["patient_id"],
            "provider_id":     "PRV-022",
            "procedure_code":  proc[0],
            "procedure_desc":  proc[1],
            "diagnosis_code":  random.choice(DIAGNOSIS_CODES),
            "date_of_service": rand_date().strftime("%Y-%m-%d"),
            "amount_billed":   round(amt, 2),
            "amount_paid":     round(amt * 0.80, 2),
            "claim_status":    "Approved",
            "is_fraud":        True,
            "fraud_type":      "UPCODING",
        })
        ctr += 1
    return rows, ctr

# ── Fraud 5: Duplicate Billing ────────────────────────────────────────────────
def generate_duplicate_billing(patients: pd.DataFrame, ctr: int):
    """PRV-055 re-submits the same claim 1–2 extra times with slight amount jitter."""
    pool = patients[patients["patient_id"].isin({pat_id(i) for i in range(200, 215)})]
    rows = []
    for _, pat in pool.iterrows():
        proc = random.choices(PROCEDURE_CODES[:6], weights=PROC_WEIGHTS[:6], k=1)[0]
        svc  = rand_date()
        diag = random.choice(DIAGNOSIS_CODES)
        amt  = proc[2] * random.uniform(0.9, 1.1)
        base = {
            "claim_id": clm_id(ctr), "patient_id": pat["patient_id"],
            "provider_id": "PRV-055", "procedure_code": proc[0],
            "procedure_desc": proc[1], "diagnosis_code": diag,
            "date_of_service": svc.strftime("%Y-%m-%d"),
            "amount_billed": round(amt, 2), "amount_paid": round(amt * 0.80, 2),
            "claim_status": "Approved", "is_fraud": False, "fraud_type": None,
        }
        rows.append(base)
        ctr += 1
        for _ in range(random.randint(1, 2)):
            dup = base.copy()
            dup["claim_id"]      = clm_id(ctr)
            dup["amount_billed"] = round(amt * random.uniform(0.97, 1.03), 2)
            dup["is_fraud"]      = True
            dup["fraud_type"]    = "DUPLICATE_BILLING"
            rows.append(dup)
            ctr += 1
    return rows, ctr

# ── Referrals ──────────────────────────────────────────────────────────────────
def generate_referrals(claims: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    """
    Normal providers refer randomly.
    The fraud ring (PRV-031/032/033) refers ONLY within itself — zero external edges.
    """
    RING       = {"PRV-031", "PRV-032", "PRV-033"}
    normal_ids = list(providers[~providers["provider_id"].isin(RING)]["provider_id"])
    ring_pats  = list(claims[claims["fraud_type"] == "REFERRAL_RING"]["patient_id"].unique())
    rows = []
    ctr  = 1

    # 400 legitimate referrals
    for _ in range(400):
        a, b = random.sample(normal_ids, 2)
        pool = claims[claims["provider_id"] == a]["patient_id"]
        p    = pool.sample(1).iloc[0] if len(pool) > 0 else pat_id(random.randint(1, 294))
        rows.append({"referral_id": ref_id(ctr), "referring_provider_id": a,
                     "referred_provider_id": b, "patient_id": p,
                     "referral_date": rand_date().strftime("%Y-%m-%d")})
        ctr += 1

    # 150 ring-only referrals
    ring_list = list(RING)
    for _ in range(150):
        a = random.choice(ring_list)
        b = random.choice([x for x in ring_list if x != a])
        p = random.choice(ring_pats) if ring_pats else pat_id(random.randint(50, 64))
        rows.append({"referral_id": ref_id(ctr), "referring_provider_id": a,
                     "referred_provider_id": b, "patient_id": p,
                     "referral_date": rand_date().strftime("%Y-%m-%d")})
        ctr += 1

    return pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating patients …")
    patients = generate_patients()

    print("Generating providers …")
    providers = generate_providers()

    print("Generating claims …")
    ctr: int  = 1
    all_rows: list = []

    rows, ctr = generate_normal_claims(patients, providers, ctr);  all_rows.extend(rows)
    rows, ctr = generate_ghost_billing(patients, ctr);             all_rows.extend(rows)
    rows, ctr = generate_referral_ring(patients, ctr);             all_rows.extend(rows)
    rows, ctr = generate_impossible_travel(patients, ctr);         all_rows.extend(rows)
    rows, ctr = generate_upcoding(patients, ctr);                  all_rows.extend(rows)
    rows, ctr = generate_duplicate_billing(patients, ctr);         all_rows.extend(rows)

    claims = pd.DataFrame(all_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Generating referrals …")
    referrals = generate_referrals(claims, providers)

    patients.to_csv(os.path.join(out_dir, "patients.csv"),      index=False)
    providers.to_csv(os.path.join(out_dir, "providers.csv"),    index=False)
    claims.to_csv(os.path.join(out_dir, "claims.csv"),          index=False)
    referrals.to_csv(os.path.join(out_dir, "referrals.csv"),    index=False)
    (claims.drop(columns=["is_fraud", "fraud_type"])
           .to_csv(os.path.join(out_dir, "claims_public.csv"), index=False))

    fraud = claims[claims["is_fraud"]]
    print(f"\nDone.")
    print(f"  Patients:  {len(patients):>6,}")
    print(f"  Providers: {len(providers):>6,}")
    print(f"  Claims:    {len(claims):>6,}  "
          f"({len(fraud)} fraudulent, {len(fraud)/len(claims)*100:.1f}%)")
    print(f"  Referrals: {len(referrals):>6,}")
    print(f"\n  Total billed: ${claims['amount_billed'].sum():>12,.2f}")
    print(f"  Fraud billed: ${fraud['amount_billed'].sum():>12,.2f}")
    print(f"\n  Breakdown:")
    for ftype, grp in fraud.groupby("fraud_type"):
        print(f"    {ftype:<25} {len(grp):>4} claims  ${grp['amount_billed'].sum():>10,.2f}")
    print(f"\n  Labelled    → data/claims.csv         (instructor only)")
    print(f"  Unlabelled  → data/claims_public.csv  (distribute to participants)")


if __name__ == "__main__":
    main()
