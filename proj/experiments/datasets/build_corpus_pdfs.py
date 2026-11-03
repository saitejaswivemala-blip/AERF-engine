"""
Builds PDF files for the experimental corpus from REAL text retrieved from
open-access sources during this session (see paper_manifest.json for exact
source URLs, license, and access-type notes per paper).

This is NOT a general-purpose PDF fetcher. It embeds verbatim excerpts of
real abstracts/results/discussion sections that were retrieved via web
search / web fetch in the authoring session, because this sandbox has no
network route to journal/preprint servers for automated fetching (see
experiments/reports/candidate_paper_selection_report.md, "Reproducibility"
section). For P1-P3, the excerpts are drawn from full papers that were
successfully retrieved. For P4-P5, only the indexed abstract was
retrievable in this session (PMC blocked automated fetch); this is stated
explicitly in each generated PDF and in the manifest.

Run: python experiments/datasets/build_corpus_pdfs.py
"""
from __future__ import annotations
import json
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "paper_manifest.json").read_text())

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=8)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceAfter=6)

# Real text content, excerpted from the full-text/abstract retrieved in this
# session. Kept to abstract + key results + relevant discussion for length;
# not the complete original document.
CONTENT = {
    "P1": {
        "sections": [
            ("Abstract", (
                "Background: A 2021 meta-analysis of 37 randomised controlled trials (RCTs) of vitamin D "
                "supplementation for prevention of acute respiratory infections (ARI) revealed a statistically "
                "significant protective effect of the intervention (odds ratio [OR] 0.92, 95% confidence interval "
                "[CI] 0.86 to 0.99). Since then, 6 eligible RCTs have completed, including one large trial "
                "(n=15,804).\n\n"
                "Findings: We identified 6 new RCTs (19,337 participants). Data were combined with data from "
                "48,488 participants in 43 previously identified RCTs, bringing the total to 64,086 participants "
                "in 46 trials. For the primary comparison of any vitamin D vs. placebo, the intervention did not "
                "significantly affect overall ARI risk (OR 0.94, 95% CI 0.88 to 1.00, P=0.057; 40 studies; "
                "I2 26.4%). Pre-specified subgroup analysis did not reveal evidence of effect modification by "
                "age, baseline vitamin D status, or dosing regimen.\n\n"
                "Interpretation: This updated meta-analysis yielded a similar point estimate for the overall "
                "effect of vitamin D supplementation on ARI risk to that obtained previously, but the 95% CI for "
                "this effect now spans 1.00, indicating no statistically significant protection."
            )),
            ("Key Subgroup Results (Table 2)", (
                "Overall (40 studies, 61,589 participants): OR 0.94 (95% CI 0.88 to 1.002).\n"
                "Age 1.00-15.99 years (16 studies): OR 0.74 (95% CI 0.60 to 0.92) - statistically significant "
                "protective effect.\n"
                "Age 16.00-64.99 years (22 studies): OR 0.96 (95% CI 0.85 to 1.08) - not significant.\n"
                "Age >=65.00 years (17 studies): OR 0.96 (95% CI 0.90 to 1.02) - not significant.\n"
                "Dosing frequency, daily (21 studies): OR 0.84 (95% CI 0.73 to 0.97) - statistically significant "
                "protective effect.\n"
                "Dosing frequency, weekly (7 studies): OR 0.97 (95% CI 0.88 to 1.06) - not significant.\n"
                "Dosing frequency, monthly or less (12 studies): OR 0.98 (95% CI 0.93 to 1.03) - not significant.\n"
                "Daily dose equivalent 400-1000 IU (10 studies): OR 0.70 (95% CI 0.55 to 0.89) - statistically "
                "significant protective effect."
            )),
            ("Discussion (excerpt)", (
                "The point estimate of the overall effect of vitamin D supplementation on ARI risk obtained in "
                "the current analysis (0.94) is similar to that yielded by our previous meta-analysis (0.92). "
                "However, in contrast to our previous work, the 95% CI for this effect now spans 1.00. Although "
                "statistically significant protection was seen within some trial subsets (daily dosing trials, "
                "trials that administered 400-1000 IU/day, trials conducted for 12 months or less and trials in "
                "participants aged 1.00-15.99 years), meta-regression analysis did not yield formal evidence to "
                "suggest that effects of vitamin D were modified by any of these factors."
            )),
        ],
    },
    "P2": {
        "sections": [
            ("Abstract", (
                "Background: Recent meta-analyses concluded that vitamin D supplementation can prevent acute "
                "respiratory infection (ARI). However, the findings were heavily influenced by results from two "
                "arms of a six-arm cluster-randomised trial that were analysed without accounting for the "
                "cluster trial design. We have used publicly available data to provide results from the "
                "remaining four unpublished trial arms and to reanalyse the meta-analyses, accounting for the "
                "cluster trial design.\n\n"
                "Results: There were 744 trial participants (6 treatment groups, 21 clusters, mean cluster "
                "size 35.4). In analyses based on summary statistics, there was no statistically significant "
                "effect of vitamin D on ARI risk in any individual treatment group, or when groups were pooled "
                "(RR 0.75, 95% CI 0.50-1.13). In individual patient data analyses, there was also no "
                "statistically significant effect of vitamin D on the ARI risk or number of ARIs in any "
                "treatment group, or when pooled: odds ratio 0.58 (0.26-1.29), rate ratio 0.70 (0.44-1.12), "
                "respectively. Update of the previous meta-analysis showed no effect of vitamin D on ARI either "
                "when data from the two arms of the trial, or when all trial arms were incorporated (RR 0.98, "
                "0.96-1.00, P=0.10 both analyses).\n\n"
                "Conclusions: Overall, vitamin D supplementation had no effect on the risk of an ARI or on the "
                "number of ARIs in this trial or in a re-analysis of the most recent meta-analysis. The results "
                "of the updated meta-analysis do not suggest that vitamin D supplementation would reduce the "
                "risk of Covid 19."
            )),
            ("Introduction (excerpt)", (
                "Two recent high profile meta-analyses from the same research group have concluded that vitamin "
                "D supplements can reduce the risk of ARI. The results of both meta-analyses are heavily "
                "influenced by the Blue Sky study, a cluster-randomised trial of vitamin D in schoolchildren "
                "from Mongolia. One example of the influence of this trial report is that in the first "
                "meta-analysis (using individual patient data), daily or weekly vitamin D reduced the odds of "
                "ARI by 70% in the subgroup with baseline 25-hydroxyvitamin D (25OHD) <25 nmol/L. Of the 234 "
                "individuals in this analysis, 192 (92%) were from the Blue Sky study. In both meta-analyses "
                "the Blue Sky trial data were analysed as though the trial was individually randomised rather "
                "than cluster-randomised. This approach is incorrect as it over-estimates the precision of the "
                "effect size."
            )),
            ("Discussion (excerpt)", (
                "These newly reported results show that vitamin D supplementation had no statistically "
                "significant effect on the proportion of individuals with an ARI or on the number of ARIs in "
                "this trial. When these data are used to update the latest trial-level meta-analysis, the "
                "pooled results for vitamin D are neutral with the confidence intervals including unity. As of "
                "February 2022, the two meta-analyses have been cited 48 and 881 times respectively, in Scopus. "
                "Many citing documents cite these two meta-analyses as evidence that vitamin D supplements may "
                "reduce the risk of Covid-19 or mitigate the illness. The updated pooled trial-level "
                "meta-analysis results do not support suggestions that vitamin D supplementation should be "
                "prescribed to prevent Covid-19."
            )),
        ],
    },
    "P3": {
        "sections": [
            ("Abstract", (
                "OBJECTIVES: To determine whether population-level implementation of a test-and-treat approach "
                "to correction of sub-optimal vitamin D status (25(OH)D <75 nmol/L) influences risk of "
                "all-cause acute respiratory infection (ARI) or coronavirus disease 2019 (COVID-19).\n\n"
                "DESIGN: Phase 3 open-label randomised controlled trial (CORONAVIT).\n\n"
                "PARTICIPANTS: 6200 adults aged 16 years or older, who were not already taking vitamin D "
                "supplements at baseline.\n\n"
                "RESULTS: Of 3100 participants offered 25(OH)D testing, 2958 (95.4%) accepted, and 2690 (86.8%) "
                "had 25(OH)D <75 nmol/L and were sent vitamin D supplements. 76 (5.0%) vs. 87 (5.7%) vs. 136 "
                "(4.6%) participants in higher-dose vs. lower-dose vs. no offer groups experienced at least one "
                "ARI of any cause (odds ratio [OR] for higher-dose vs. no offer 1.09, 95% CI 0.82-1.46; "
                "lower-dose vs. no offer 1.26, 0.96-1.66). 45 (3.0%) vs. 55 (3.6%) vs. 78 (2.6%) participants "
                "developed COVID-19 (OR for higher-dose vs. no offer 1.13, 0.78-1.63; lower-dose vs. no offer "
                "1.39, 0.98-1.97).\n\n"
                "CONCLUSIONS: Among adults with a high baseline prevalence of sub-optimal vitamin D status, "
                "implementation of a population-level test-and-treat approach to vitamin D replacement did not "
                "reduce risk of all-cause ARI or COVID-19."
            )),
            ("Discussion (excerpt)", (
                "No statistically significant effect of either dose was seen on the primary outcome of incident "
                "doctor- or swab test-confirmed ARI, or on the major secondary outcome of incident swab "
                "test-confirmed COVID-19. Oral vitamin D supplementation was safe and well-tolerated at both "
                "doses investigated. The results from the current study do not support the hypothesis that such "
                "regimens offer protection against ARI, and are consistent with those of several other recent "
                "phase 3 trials of vitamin D supplementation that have reported no effect of vitamin D "
                "supplementation on risk of ARIs. This contrasts with recent findings from a Phase 2 RCT "
                "conducted in Mexican health care workers, which reported a strong protective effect of a daily "
                "dose of 4,000 IU vitamin D."
            )),
        ],
    },
    "P4": {
        "sections": [
            ("Note on source", (
                "This PDF was assembled from the indexed abstract of PMCID PMC8955452 only. The PMC article "
                "page could not be automatically retrieved in this authoring session (blocked by an automated "
                "bot-check interstitial); the full text is confirmed to be freely available at "
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC8955452/ and should be substituted for this abstract "
                "excerpt before treating any experimental result involving this paper as final."
            )),
            ("Abstract (as indexed)", (
                "Efficacy of Vitamin D Supplements in Treatment of Acute Respiratory Infection: A Meta-analysis "
                "for Randomized Controlled Trials. PMCID: PMC8955452, PMID: 35334804.\n\n"
                "Background: Recent randomized controlled trials (RCTs) have reported inconsistent findings "
                "regarding the efficacy of vitamin D supplementation in the treatment of acute respiratory "
                "infections (ARIs). This study aimed to investigate the efficacy of vitamin D supplementation "
                "in the treatment of ARIs using a meta-analysis of RCTs.\n\n"
                "Methods: PubMed, EMBASE, and the Cochrane Library were searched for relevant articles in June "
                "2021. Results: Out of 390 articles retrieved from the databases, we included 18 RCTs, which "
                "involved 3648 participants, with 1838 in an intervention group and 1810 in a control group in "
                "the final analysis. In the meta-analysis of all the trials, vitamin D supplements had a "
                "beneficial effect in the treatment of ARIs (relative risk (RR) = 1.07; 95% confidence interval "
                "(CI), 1.01-1.13; I2 = 66.9%)."
            )),
        ],
    },
    "P5": {
        "sections": [
            ("Note on source", (
                "This PDF was assembled from the indexed abstract of PMCID PMC12699866 only. The PMC article "
                "page could not be automatically retrieved in this authoring session (blocked by an automated "
                "bot-check interstitial); the full text is confirmed to be freely available at "
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC12699866/ and should be substituted for this abstract "
                "excerpt before treating any experimental result involving this paper as final."
            )),
            ("Abstract (as indexed)", (
                "The role of vitamin D in the prevention and treatment of acute respiratory infections in "
                "pediatric populations: a systematic review and meta-analysis of randomized controlled trials. "
                "PMCID: PMC12699866.\n\n"
                "A systematic review and meta-analysis of randomized controlled trials (RCTs) was conducted in "
                "accordance with PRISMA guidelines. Eligible RCTs enrolled pediatric populations, comparing "
                "oral vitamin D supplementation with placebo or no treatment, and reported ARI-related "
                "outcomes. Seventeen RCTs including 18,372 participants were analyzed.\n\n"
                "Overall, vitamin D supplementation did not significantly reduce ARI incidence (RR = 0.82, 95% "
                "CI: 0.64 to 1.06), hospitalization duration (MD = 0.03 days, 95% CI: -0.72 to 0.78), recovery "
                "time (MD = -4.36 h, 95% CI: -11.87 to 3.15), or all-cause mortality (RR = 0.88, 95% CI: 0.60 "
                "to 1.28)."
            )),
        ],
    },
}


def build_pdf(paper_id: str) -> Path:
    meta = next(p for p in MANIFEST["papers"] if p["paper_id"] == paper_id)
    out_path = HERE.parent.parent / meta["pdf_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER,
                             leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [
        Paragraph(meta["title"], h1),
        Paragraph(f"{meta['authors']} ({meta['year']}). {meta['venue']}. DOI/ID: {meta['doi']}", body),
        Paragraph(f"Source: {meta['source_url']}", body),
        Spacer(1, 12),
    ]
    for heading, text in CONTENT[paper_id]["sections"]:
        story.append(Paragraph(heading, h2))
        for para in text.split("\n\n"):
            story.append(Paragraph(para.replace("\n", "<br/>"), body))
        story.append(Spacer(1, 6))
    doc.build(story)
    return out_path


if __name__ == "__main__":
    for pid in CONTENT:
        p = build_pdf(pid)
        print(f"Built {pid}: {p} ({p.stat().st_size} bytes)")
