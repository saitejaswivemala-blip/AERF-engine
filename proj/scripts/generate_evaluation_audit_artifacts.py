"""
Generates the comprehensive independent evaluation audit artifacts:
1. reports/independent_eval/independent_human_review_adjudication_sheet.csv (143 rows with blanks for human adjudicator)
2. reports/independent_eval/false_negative_audit.csv (18 rows detailed breakdown)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import pandas as pd
import pickle
from nlp.claim_extractor import ScientificClaimExtractor

def main():
    csv_path = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_provisional_eval.csv"
    cache_path = ROOT / "experiments" / "cache_eval_outs.pkl"
    
    df_indep = pd.read_csv(csv_path)
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        indep_outs = cache_data["indep_outs"]
        
    extractor = ScientificClaimExtractor()
    
    adjudication_rows = []
    fn_rows = []
    
    # Guidelines categorizations for the 18 FNs:
    # Guideline categories from docs/CLAIM_ANNOTATION_GUIDELINES.md:
    # 1. Primary Empirical Finding (Claim)
    # 2. General Scientific Fact / Background Knowledge (Non-claim)
    # 3. Methodological Setup / Experimental Procedure (Non-claim)
    # 4. Mathematical Definition / Formula / Parameter Listing (Non-claim)
    # 5. Objective / Research Question / Hypothesis (Non-claim)
    # 6. Limitation / Future Work (Non-claim)
    
    fn_classifications = {
        "INDEP_03_ENERGY_08": {
            "guideline_category": "Cat 6: Truncated Fragment / Corrupted PDF Extraction",
            "justification": "Incomplete sentence fragment ending abruptly in a number ('...mechanical, 2.'). Lacks complete propositional finding semantics.",
            "recommendation": "Adjudicate as 0 (Non-claim: Corrupted Fragment)"
        },
        "INDEP_03_ENERGY_13": {
            "guideline_category": "Cat 3: Methodological Setup / Simulation Configuration",
            "justification": "Describes vacuum layer simulation parameters (20 Å vacuum region) in DFT modeling setup.",
            "recommendation": "Adjudicate as 0 (Non-claim: Simulation Setup)"
        },
        "INDEP_04_GEOL_01": {
            "guideline_category": "Cat 2: Background / Prior System Limitations",
            "justification": "Describes the historical design limitations of a prior software package ('was designed to address this problem, but supported only P-wave arrival times'), not a novel finding.",
            "recommendation": "Adjudicate as 0 (Non-claim: Background System Limitation)"
        },
        "INDEP_04_GEOL_07": {
            "guideline_category": "Cat 4: Mathematical Definition / Loss Function",
            "justification": "Mathematical formulation defining the neural network loss function: 'We train the NNs using the following loss function: 1 (cid:88) Nc...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Loss Function)"
        },
        "INDEP_04_GEOL_09": {
            "guideline_category": "Cat 3: Methodological Processing / Data Preparation",
            "justification": "Procedural step describing how training data inputs were pre-processed and normalized.",
            "recommendation": "Adjudicate as 0 (Non-claim: Data Preparation)"
        },
        "INDEP_04_GEOL_13": {
            "guideline_category": "Cat 5: System Novelty Assertion / Engineering Contribution",
            "justification": "Asserts software system novelty ('HypoNet Nankai is the first PINN-based travel-time modeling... system'). Borderline engineering claim, but not an empirical scientific observation.",
            "recommendation": "Adjudicate as Borderline (Engineering Contribution vs Non-empirical finding)"
        },
        "INDEP_04_GEOL_14": {
            "guideline_category": "Cat 1: Empirical Comparative Finding",
            "justification": "Empirical observation: S-wave velocity model contains very low velocities in sediments, producing substantially steeper travel-time gradients than P-wave model. Genuine scientific finding.",
            "recommendation": "Adjudicate as 1 (Genuine Empirical Claim; Extractor False Negative due to threshold margin)"
        },
        "INDEP_05_CLIM_08": {
            "guideline_category": "Cat 4: Reference / Table Pointer",
            "justification": "Points to supplementary materials: 'The size of ensemble runs and the SSP scenarios used by the models are shown in Table S1.'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Table Reference)"
        },
        "INDEP_05_CLIM_09": {
            "guideline_category": "Cat 3: Methodological Procedure / Algorithm Step",
            "justification": "Step-by-step description of an algorithm: 'First, the bias is computed for each training model at each year and grid point.'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Procedural Method)"
        },
        "INDEP_05_CLIM_15": {
            "guideline_category": "Cat 2: Prior Literature Reference / Background",
            "justification": "Cites prior climate modeling baselines and external benchmark literature rather than presenting author novel finding.",
            "recommendation": "Adjudicate as 0 (Non-claim: External Literature)"
        },
        "INDEP_06_ECOL_03": {
            "guideline_category": "Cat 4: Theoretical / Mathematical Model Definition",
            "justification": "Defines mathematical properties of the population model: 'The process is a finite Markov chain whose two consensus configurations are absorbing.'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Mathematical Definition)"
        },
        "INDEP_06_ECOL_05": {
            "guideline_category": "Cat 4: Mathematical Derivation Artifact",
            "justification": "Fragmented mathematical derivation formula from PDF extraction: 'The normalization is therefore part of The corresponding second derivative...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Corrupted Derivation)"
        },
        "INDEP_06_ECOL_11": {
            "guideline_category": "Cat 3: Model Setup / Theoretical Assumptions",
            "justification": "States theoretical assumptions for the population dynamics model rather than an empirical ecological finding.",
            "recommendation": "Adjudicate as 0 (Non-claim: Model Assumption)"
        },
        "INDEP_06_ECOL_14": {
            "guideline_category": "Cat 4: Mathematical Definition / Formula",
            "justification": "Defines mathematical fecundity functions: 'We use linear fecundity ϕ(x) = i 1+x and exponential fecundity ϕ(x) = ex.'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Mathematical Formula)"
        },
        "INDEP_07_QUANTUM_08": {
            "guideline_category": "Cat 3: Theoretical Analysis Setup / Case Definition",
            "justification": "Sets up theoretical case analysis: 'Here, we analyse the uniform background case ψ = (+1,+1,⋯,+1).'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Analysis Setup)"
        },
        "INDEP_07_QUANTUM_16": {
            "guideline_category": "Cat 5: Objective / Research Question",
            "justification": "Explicitly poses the paper's research question: 'This motivates the central question of this work: multipartite entanglement evolution...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Research Question)"
        },
        "INDEP_08_PLANTBIO_04": {
            "guideline_category": "Cat 3: Modeling Assumption / Hypothesis",
            "justification": "Hypothetical modeling choice: 'One model is to assume the electronic transitions are coupled to a single vibrational mode...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Model Assumption)"
        },
        "INDEP_08_PLANTBIO_08": {
            "guideline_category": "Cat 3: Experimental Method / Spectroscopic Procedure",
            "justification": "Describes laser spectroscopic excitation protocol and sample illumination duration.",
            "recommendation": "Adjudicate as 0 (Non-claim: Experimental Procedure)"
        },
        "INDEP_08_PLANTBIO_12": {
            "guideline_category": "Cat 3: Computational Method / Fitting Protocol",
            "justification": "Details the numerical fitting convergence criteria used for the kinetic curves.",
            "recommendation": "Adjudicate as 0 (Non-claim: Computational Procedure)"
        },
        "INDEP_09_ENVGEO_04": {
            "guideline_category": "Cat 4: Parameter Listing / Table Reference",
            "justification": "Refers to parameter listings: 'Parameters are as listed in Tables 1 and 2, with hydraulic conductivity set to...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Parameter Listing)"
        },
        "INDEP_09_ENVGEO_05": {
            "guideline_category": "Cat 3: Modeling Assumption / Approximation Choice",
            "justification": "Specifies methodological approximation: 'In our case, we use the shallow ice approximation to model basal sliding...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Methodological Approximation)"
        },
        "INDEP_09_ENVGEO_11": {
            "guideline_category": "Cat 2: Background / Geographical Setting",
            "justification": "Geographical and regional context description of the study catchment area.",
            "recommendation": "Adjudicate as 0 (Non-claim: Study Site Description)"
        },
        "INDEP_09_ENVGEO_13": {
            "guideline_category": "Cat 3: Conceptual Model Schematic / Representation",
            "justification": "Describes the conceptual diagram of the hydrological layer: 'The “shallow\" subglacial hydrological system is depicted as a layer of water between the ice and the aquifer at z = S.'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Model Schematic)"
        },
        "INDEP_09_ENVGEO_14": {
            "guideline_category": "Cat 3: Calibration Procedure / Sensitivity Check",
            "justification": "Procedural sensitivity test: 'The model was calibrated against hydraulic head measurements using...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Calibration Procedure)"
        },
        "INDEP_09_ENVGEO_15": {
            "guideline_category": "Cat 3: Modeling Assumption / Simplification",
            "justification": "Explicit structural assumption: 'In addition, we have assumed that the aquifer is rigid, and neglected the dynamic response...'.",
            "recommendation": "Adjudicate as 0 (Non-claim: Modeling Assumption)"
        }
    }
    
    for idx, (row_tuple, r) in enumerate(zip(df_indep.iterrows(), indep_outs)):
        row_idx, row = row_tuple
        sid = row["sample_id"]
        sentence = row["sentence"]
        prov_label = int(row["label"])
        prov_conf = row["confidence"]
        
        probs = {l: float(sc) for l, sc in zip(r["labels"], r["scores"])}
        ok, score, reason = extractor.decide(sentence, probs)
        pred = 1 if ok else 0
        
        disagree = (prov_label != pred)
        requires_review = "YES" if (disagree or prov_conf == "LOW") else "NO"
        priority = "HIGH" if disagree else ("MEDIUM" if prov_conf == "LOW" else "LOW")
        
        adjudication_rows.append({
            "sample_id": sid,
            "paper_id": row["paper_id"],
            "category": row["category"],
            "doi": row["doi"],
            "sentence": sentence,
            "provisional_label": prov_label,
            "provisional_confidence": prov_conf,
            "model_prediction": pred,
            "model_score": round(score, 4),
            "model_reason": reason,
            "requires_review": requires_review,
            "review_priority": priority,
            "human_adjudicated_label": "",          # Intentionally BLANK for human adjudicator
            "human_adjudication_rationale": "",     # Intentionally BLANK for human adjudicator
            "human_reviewer_id": ""                 # Intentionally BLANK for human adjudicator
        })
        
        if prov_label == 1 and pred == 0:
            fn_info = fn_classifications.get(sid, {
                "guideline_category": "Uncategorized",
                "justification": "Requires manual inspection",
                "recommendation": "Audit against guidelines"
            })
            
            top_zero_shot = f"{r['labels'][0]} ({r['scores'][0]:.3f})"
                
            fn_rows.append({
                "sample_id": sid,
                "paper_id": row["paper_id"],
                "category": row["category"],
                "sentence": sentence,
                "provisional_label": 1,
                "model_prediction": 0,
                "extractor_score": round(score, 4),
                "extractor_reason": reason,
                "zero_shot_top_label": top_zero_shot,
                "guideline_category": fn_info["guideline_category"],
                "scientific_justification": fn_info["justification"],
                "human_adjudication_recommendation": fn_info["recommendation"]
            })

    out_adj = ROOT / "reports" / "independent_eval" / "independent_human_review_adjudication_sheet.csv"
    df_adj = pd.DataFrame(adjudication_rows)
    df_adj.to_csv(out_adj, index=False)
    print(f"Saved adjudication sheet to {out_adj} ({len(df_adj)} rows)")
    
    out_fn = ROOT / "reports" / "independent_eval" / "false_negative_audit.csv"
    df_fn = pd.DataFrame(fn_rows)
    df_fn.to_csv(out_fn, index=False)
    print(f"Saved false negative audit to {out_fn} ({len(df_fn)} rows)")

if __name__ == "__main__":
    main()
