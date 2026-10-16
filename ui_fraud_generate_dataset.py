"""
generate_dataset.py
--------------------
Generates a synthetic but realistic HEALTH INSURANCE CLAIM dataset for the
"Predicting Insurance Claim Fraud" project, using the University of Ibadan
(UI) as the case study.

Case study grounding
---------------------
The University of Ibadan runs its own health service, the University
Health Service (popularly known as "Jaja Clinic"), which provides care to
staff, staff dependants, students, and National Health Insurance Scheme
(NHIS) enrollees, processing several hundred visits a day. A published
study (Owumi, Omorogbe & Raphael, 2013, African Journal of Social
Sciences) specifically examined the NHIS scheme's impact on University of
Ibadan employees, confirming that a functioning claims/insurance process
has existed at UI for over a decade. The Clinic also publicly references
a health insurance scheme with an annual cost structure (e.g. a
multi-person annual plan), which is used below to calibrate a realistic
annual claim ceiling.

No real UI/Jaja Clinic claim-level records are public (health records are
confidential), so a synthetic dataset is generated instead, encoding fraud
indicators that are well documented in health-insurance fraud detection
research: upcoding (billing for a more expensive service than was given),
diagnosis-treatment mismatches, phantom/duplicate billing across
providers, missing referral documentation for specialist care, and
suspicious timing (claims filed on weekends/holidays when oversight is
thinner).

Output: data/ui_health_insurance_claims.csv (5000 claim records)
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 5000  # number of claim records

# ---------------------------------------------------------------
# 1. Beneficiary and enrollment attributes
# ---------------------------------------------------------------
beneficiary_types = ["Academic Staff", "Non-Teaching Staff", "Staff Dependant",
                     "Student (NHIS)", "Postgraduate Student"]
beneficiary_type = rng.choice(beneficiary_types, N, p=[0.22, 0.28, 0.20, 0.20, 0.10])

years_registered = np.clip(rng.exponential(3.5, N), 0.1, 25).round(1)

providers = ["Jaja Clinic (UHS)", "UCH Referral", "Private Accredited Hospital"]
provider = rng.choice(providers, N, p=[0.62, 0.23, 0.15])

# ---------------------------------------------------------------
# 2. Visit / claim attributes
# ---------------------------------------------------------------
visit_types = ["Consultation", "Laboratory Test", "Pharmacy/Drugs",
               "Specialist Referral", "Admission"]
visit_type = rng.choice(visit_types, N, p=[0.38, 0.22, 0.20, 0.13, 0.07])

diagnosis_categories = ["Malaria", "Typhoid", "Respiratory Infection",
                         "Hypertension/Diabetes", "Maternity", "Minor Surgery", "Other"]
diagnosis_category = rng.choice(diagnosis_categories, N,
                                 p=[0.28, 0.16, 0.15, 0.14, 0.10, 0.07, 0.10])

admission_days = np.where(
    visit_type == "Admission",
    np.clip(rng.exponential(2.5, N), 1, 21).round(0),
    0
)

# Annual claim ceiling calibrated to Jaja Clinic's publicised scheme cost
# structure (an approx. NGN 45,000 / year multi-person plan implies a
# typical per-person annual ceiling in the range used below).
annual_claim_ceiling_NGN = 40_000

# Days since the beneficiary's previous claim (short gaps can indicate
# frequent/phantom re-billing)
days_since_last_claim = np.clip(rng.exponential(35, N), 0, 365).round(0)

prior_claims_this_year = rng.poisson(1.4, N)

referral_letter_provided = np.where(
    visit_type.__eq__("Specialist Referral"),
    rng.choice([0, 1], N, p=[0.35, 0.65]),
    1  # not applicable / assumed fine for non-referral visit types
)

multiple_providers_same_week = rng.choice([0, 1], N, p=[0.88, 0.12])
weekend_or_holiday_claim = rng.choice([0, 1], N, p=[0.82, 0.18])

# Diagnosis-treatment mismatch: a rough proxy for upcoding, e.g. a claim
# for "Admission" or "Specialist Referral" attached to a mild diagnosis
# like Malaria/Typhoid with no admission days, or vice versa.
mismatch_risk = (
    ((visit_type == "Admission") & (admission_days <= 1) &
     np.isin(diagnosis_category, ["Malaria", "Typhoid"]))
    | ((visit_type == "Specialist Referral") & (referral_letter_provided == 0))
)
diagnosis_treatment_mismatch = (
    mismatch_risk | (rng.random(N) < 0.05)  # small baseline noise rate
).astype(int)

# ---------------------------------------------------------------
# 3. Claim amount
#    Driven by visit type / admission length, with realistic noise.
# ---------------------------------------------------------------
base_cost = pd.Series(visit_type).map({
    "Consultation": 2500, "Laboratory Test": 4500, "Pharmacy/Drugs": 6000,
    "Specialist Referral": 9000, "Admission": 15000,
}).to_numpy()

claim_amount_NGN = (
    base_cost * (1 + 0.35 * admission_days) * rng.normal(1.0, 0.25, N)
)
claim_amount_NGN = np.clip(claim_amount_NGN, 800, 250_000).round(-2)

claim_to_ceiling_ratio = (claim_amount_NGN / annual_claim_ceiling_NGN).round(3)

# ---------------------------------------------------------------
# 4. Latent fraud risk score -> fraud label
#    Each term encodes a documented health-insurance fraud red flag.
# ---------------------------------------------------------------
risk = (
    -3.35                                                          # base rate control
    + 1.35 * diagnosis_treatment_mismatch                          # upcoding proxy
    + 1.10 * (referral_letter_provided == 0)                       # missing documentation
    + 1.20 * multiple_providers_same_week                          # duplicate/phantom billing
    + 0.75 * weekend_or_holiday_claim                               # thinner oversight
    + 0.55 * np.clip(claim_to_ceiling_ratio - 1.0, 0, 5)            # exceeds typical ceiling
    + 0.40 * prior_claims_this_year                                 # heavy repeat utilisation
    + 0.85 * (days_since_last_claim < 3)                            # implausibly rapid re-claim
    + 0.45 * (provider == "Private Accredited Hospital")            # less directly supervised than Jaja Clinic
    - 0.05 * years_registered                                       # longer-tenured enrollees, mild protective effect
    + rng.normal(0, 0.45, N)                                        # unexplained variation
)

fraud_prob = 1 / (1 + np.exp(-risk))
fraud_reported = (rng.random(N) < fraud_prob).astype(int)

# ---------------------------------------------------------------
# 5. Assemble and save
# ---------------------------------------------------------------
df = pd.DataFrame({
    "claim_id": [f"UIJC{200000 + i}" for i in range(N)],
    "beneficiary_type": beneficiary_type,
    "years_registered": years_registered,
    "provider": provider,
    "visit_type": visit_type,
    "diagnosis_category": diagnosis_category,
    "admission_days": admission_days.astype(int),
    "claim_amount_NGN": claim_amount_NGN.astype(int),
    "claim_to_ceiling_ratio": claim_to_ceiling_ratio,
    "prior_claims_this_year": prior_claims_this_year,
    "days_since_last_claim": days_since_last_claim.astype(int),
    "referral_letter_provided": referral_letter_provided,
    "diagnosis_treatment_mismatch": diagnosis_treatment_mismatch,
    "multiple_providers_same_week": multiple_providers_same_week,
    "weekend_or_holiday_claim": weekend_or_holiday_claim,
    "fraud_reported": fraud_reported,
})

df.to_csv("/home/claude/fraud_ui/data/ui_health_insurance_claims.csv", index=False)

print("Saved dataset:", df.shape)
print(df.head())
print("\nFraud class balance:")
print(df.fraud_reported.value_counts())
print(f"Fraud rate: {df.fraud_reported.mean():.2%}")

