import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

in_path = ROOT / "reports/independent_eval/priority_review.csv"
out_path = ROOT / "reports/independent_eval/priority_review_decisions.csv"

df = pd.read_csv(in_path)

# Precise adjudications for all 28 samples
decisions = {
    "INDEP_02_ASTRO_12": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Spliced multi-column PDF text combining future JWST observation plans with citation. Borderline between corrupted non-claim (0) and future speculative interpretation. Needs human adjudicator confirmation."
    },
    "INDEP_03_ENERGY_08": {
        "final_label": "0",
        "rationale": "Incomplete sentence fragment from PDF parsing ending abruptly in a number ('...mechanical, 2.'). Lacks complete propositional finding semantics (Cat 11)."
    },
    "INDEP_03_ENERGY_13": {
        "final_label": "0",
        "rationale": "Describes DFT simulation setup parameters (20 Å vacuum region) rather than an empirical physical finding (Cat 6)."
    },
    "INDEP_03_ENERGY_15": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Spliced sentence text reporting improved alkali-metal adsorption via phonon dispersion calculations, but confounded by corrupted tokens. Requires human review to determine if the author's own finding is being stated."
    },
    "INDEP_04_GEOL_01": {
        "final_label": "0",
        "rationale": "Contextual description of historical design limitations of an earlier software version, not the empirical outcome of the current paper (Cat 5)."
    },
    "INDEP_04_GEOL_07": {
        "final_label": "0",
        "rationale": "Formal mathematical definition of neural network loss function and coordinate notation (Cat 10 / Cat 6)."
    },
    "INDEP_04_GEOL_13": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Asserts system engineering priority ('is the first PINN-based... system'). Borderline between an engineering contribution claim (Cat 4: label=1) and a software system description without empirical earth-science measurement (Cat 6: label=0). Needs human policy decision."
    },
    "INDEP_05_CLIM_08": {
        "final_label": "0",
        "rationale": "Pointer to supplementary table listing ensemble run sizes and SSP scenarios (Cat 6 / Cat 7)."
    },
    "INDEP_05_CLIM_09": {
        "final_label": "0",
        "rationale": "Step-by-step description of an algorithm for bias computation across models (Cat 6)."
    },
    "INDEP_06_ECOL_03": {
        "final_label": "0",
        "rationale": "Mathematical definition of Markov chain absorption properties in theoretical population modeling, not an empirical ecological outcome (Cat 10)."
    },
    "INDEP_06_ECOL_05": {
        "final_label": "0",
        "rationale": "Fragmented mathematical derivation formula from PDF extraction lacking finding semantics (Cat 10 / Cat 11)."
    },
    "INDEP_06_ECOL_06": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Asserts a dynamic effect on contact resources, but contains corrupted formula tokens ('y/t x/t ⋄ ⋄'). Requires human review on whether text corruption invalidates the claim."
    },
    "INDEP_06_ECOL_11": {
        "final_label": "0",
        "rationale": "Pure mathematical limit and asymptotic big-O derivation without empirical biological finding content (Cat 10)."
    },
    "INDEP_06_ECOL_14": {
        "final_label": "0",
        "rationale": "Formal algebraic definition of fecundity equations used in simulation setup (Cat 6 / Cat 10)."
    },
    "INDEP_07_QUANTUM_03": {
        "final_label": "1",
        "rationale": "Direct statement of empirical demonstration by the authors ('We demonstrate this crossover... the resulting states are strongly entangled'). Labeled 0 in provisional consensus due to citation brackets, but clearly represents a primary finding under Cat 1."
    },
    "INDEP_07_QUANTUM_04": {
        "final_label": "0",
        "rationale": "Figure legend caption describing visual curve and mathematical background definition (Cat 6 / Cat 11)."
    },
    "INDEP_07_QUANTUM_06": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Describes theoretical OTOC growth scaling in short times, but references 'as mentioned in the main text' and mathematical validity regime. Borderline theoretical finding vs background notation. Requires human review."
    },
    "INDEP_07_QUANTUM_08": {
        "final_label": "0",
        "rationale": "Sets up the theoretical background case for mathematical analysis rather than presenting an empirical result (Cat 6)."
    },
    "INDEP_07_QUANTUM_13": {
        "final_label": "0",
        "rationale": "Incomplete grammatical fragment referencing Eq. (24) lacking propositional assertion (Cat 11)."
    },
    "INDEP_07_QUANTUM_16": {
        "final_label": "0",
        "rationale": "Explicit statement of research question / motivation prior to empirical outcome (Cat 8)."
    },
    "INDEP_08_PLANTBIO_01": {
        "final_label": "0",
        "rationale": "Mathematical quantum mechanical formula definition of transition probabilities with formula artifacts (Cat 10)."
    },
    "INDEP_08_PLANTBIO_02": {
        "final_label": "0",
        "rationale": "Incomplete section list header fragment ending in a number (Cat 11)."
    },
    "INDEP_08_PLANTBIO_04": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Literature-cited modeling assumption in Introduction section. Borderline theoretical modeling framing vs non-claim literature citation. Needs human confirmation."
    },
    "INDEP_09_ENVGEO_04": {
        "final_label": "0",
        "rationale": "Parameter values and table reference for hydraulic conductivity (Cat 6)."
    },
    "INDEP_09_ENVGEO_05": {
        "final_label": "0",
        "rationale": "Methodological approximation choice and rheology assumption for ice sheet modeling (Cat 6)."
    },
    "INDEP_09_ENVGEO_06": {
        "final_label": "0",
        "rationale": "Mathematical formula derivation of boundary condition state and outflux equation with formula tokens (Cat 10)."
    },
    "INDEP_09_ENVGEO_13": {
        "final_label": "0",
        "rationale": "Descriptive representation of a conceptual model diagram ('is depicted as a layer...') rather than an empirical geological finding (Cat 6)."
    },
    "INDEP_09_ENVGEO_15": {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Stated modeling simplification and limitation in Discussion section. Borderline study limitation (Cat 9: label=0) vs author theoretical assertion. Needs human review."
    }
}

final_labels = []
rationales = []

for idx, r in df.iterrows():
    sid = r["sample_id"]
    dec = decisions.get(sid, {
        "final_label": "NEEDS_HUMAN_REVIEW",
        "rationale": "Requires human adjudication."
    })
    final_labels.append(dec["final_label"])
    rationales.append(dec["rationale"])

df["final_label"] = final_labels
df["decision_rationale"] = rationales

df.to_csv(out_path, index=False)
print(f"Saved decisions to {out_path} ({len(df)} rows)")
print("\nFinal label distribution:")
print(df["final_label"].value_counts())
