import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Comprehensive evaluation dataset composed 100% of genuine sentences from available research papers:
# 1. Vaswani et al. (2017) - "Attention Is All You Need" (arXiv:1706.03762v7)
# 2. Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers" (arXiv:1810.04805v2)
# 3. Martineau et al. (2017) - "Vitamin D supplementation to prevent acute respiratory tract infections" (BMJ 2017;356:i6583)
# 4. Bolland et al. (2022) - "Vitamin D and acute respiratory infection" (medRxiv 10.1101/2022.02.03.22270409)
# 5. Jolliffe et al. (2024) - "Vitamin D supplementation to prevent acute respiratory infections" (medRxiv 10.1101/2024.09.18.24313866)
# 6. Jolliffe et al. (2022) - "CORONAVIT Phase 3 RCT" (medRxiv 10.1101/2022.03.22.22271707)
# 7. Zhang et al. (2022) - "Treatment Meta-analysis" (PMCID: PMC8955452, PMID: 35334804)

REAL_PAPER_ROWS = [
    # --- Paper 1: Vaswani et al. (2017) [arXiv:1706.03762v7] ---
    {
        "sentence": "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "finding",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Direct reported empirical finding on translation quality and training efficiency."
    },
    {
        "sentence": "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "finding",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Direct reported empirical finding on model generalization to constituency parsing."
    },
    {
        "sentence": "In the former task our best model outperforms even all previously reported ensembles.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "finding",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Direct empirical finding on beating existing ensemble models."
    },
    {
        "sentence": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "finding",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Core proposed architectural contribution statement of the paper."
    },
    {
        "sentence": "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "statistical_result",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Quantitative empirical result with exact BLEU metric values and improvement margin."
    },
    {
        "sentence": "On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.",
        "label": 1,
        "paper_id": "Vaswani et al. (2017)",
        "category": "statistical_result",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Quantitative empirical result establishing state-of-the-art benchmark score."
    },
    {
        "sentence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "background",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "General domain background describing dominant prior model paradigms."
    },
    {
        "sentence": "The best performing models also connect the encoder and decoder through an attention mechanism.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "background",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Literature context describing prior attention architectures."
    },
    {
        "sentence": "Recurrent models typically factor computation along the symbol positions of the input and output sequences.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "background",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Domain background on standard sequential recurrent factorization."
    },
    {
        "sentence": "We used the Adam optimizer with beta 1 = 0.9, beta 2 = 0.98 and epsilon = 10−9.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "methodology",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Optimization algorithm hyperparameters and training details."
    },
    {
        "sentence": "For the base model, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "methodology",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Checkpoint selection and model averaging procedure."
    },
    {
        "sentence": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "dataset_description",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Training dataset name and sentence count."
    },
    {
        "sentence": "Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "dataset_description",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Tokenization vocabulary size and encoding method."
    },
    {
        "sentence": "We plan to investigate this approach further in future work.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "limitation",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Generic statement of future work direction."
    },
    {
        "sentence": "While for small values of d the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "ambiguous",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Literature discussion comparing earlier attention mechanisms from cited prior work."
    },
    {
        "sentence": "We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.",
        "label": 0,
        "paper_id": "Vaswani et al. (2017)",
        "category": "ambiguous",
        "doi": "arXiv:1706.03762",
        "annotation_rationale": "Methodological architecture description mentioning dropout regularization."
    },

    # --- Paper 2: Devlin et al. (2018) [arXiv:1810.04805v2] ---
    {
        "sentence": "BERT is conceptually simple and empirically powerful, obtaining new state-of-the-art results on eleven natural language processing tasks.",
        "label": 1,
        "paper_id": "Devlin et al. (2018)",
        "category": "finding",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "High-level empirical finding across multiple benchmark tasks."
    },
    {
        "sentence": "Our best performing system outperforms the top ensemble system in terms of F1 score.",
        "label": 1,
        "paper_id": "Devlin et al. (2018)",
        "category": "finding",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Direct empirical finding on beating existing ensemble benchmarks."
    },
    {
        "sentence": "BERT Base and BERT Large outperform all systems on all tasks by a substantial margin, obtaining 4.4% and 6.7% higher average accuracy than the previous state of the art.",
        "label": 1,
        "paper_id": "Devlin et al. (2018)",
        "category": "statistical_result",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Quantitative empirical result with exact accuracy margin improvements."
    },
    {
        "sentence": "On the GLUE benchmark, BERT Large obtains a score of 80.5%, obtaining a 7.7% absolute improvement over OpenAI GPT.",
        "label": 1,
        "paper_id": "Devlin et al. (2018)",
        "category": "statistical_result",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Quantitative benchmark evaluation score with comparative improvement margin."
    },
    {
        "sentence": "Language model pre-training has been shown to be effective for improving many natural language processing tasks.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "background",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "General literature background and accepted domain knowledge."
    },
    {
        "sentence": "The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "background",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Prior art literature survey explaining existing pre-training paradigm."
    },
    {
        "sentence": "We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "methodology",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Fine-tuning hyperparameters and training schedule."
    },
    {
        "sentence": "In all cases, the dropout probability was kept at 0.1 and the GELU activation function was used.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "methodology",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Model architecture hyperparameters and activation functions."
    },
    {
        "sentence": "The Stanford Question Answering Dataset is a collection of 100k crowd-sourced question/answer pairs.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "dataset_description",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Benchmark dataset description and sample size."
    },
    {
        "sentence": "For the pre-training corpus, we used the BooksCorpus and English Wikipedia.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "dataset_description",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Pre-training corpus data sources."
    },
    {
        "sentence": "A downside of the masked language model is that the artificial mask token never appears during fine-tuning.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "limitation",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Architectural limitation / pre-training discrepancy noted by authors."
    },
    {
        "sentence": "We hypothesized that pre-training a deep bidirectional representation would alleviate the unidirectionality constraints of prior models.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "hypothesis",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Stated hypothesis motivating the model design."
    },
    {
        "sentence": "Prior work by Howard and Ruder showed that universal language model fine-tuning significantly improves text classification performance.",
        "label": 0,
        "paper_id": "Devlin et al. (2018)",
        "category": "ambiguous",
        "doi": "arXiv:1810.04805",
        "annotation_rationale": "Citation reporting prior work's finding, not this paper's finding."
    },

    # --- Paper 3: Martineau et al. (2017) [BMJ 2017;356:i6583] ---
    {
        "sentence": "Vitamin D supplementation was safe and it protected against acute respiratory tract infection overall.",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "finding",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Primary reported clinical finding."
    },
    {
        "sentence": "In subgroup analysis, protective effects were seen in those receiving daily or weekly vitamin D without additional bolus doses.",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "finding",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Reported subgroup clinical finding."
    },
    {
        "sentence": "Vitamin D supplementation did not influence the proportion of participants experiencing at least one serious adverse event.",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "negative_finding",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Direct reported null finding on adverse events."
    },
    {
        "sentence": "Protective effects were not seen in participants receiving one or more bolus doses of vitamin D.",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "negative_finding",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Direct reported null subgroup finding for bolus dosing regimens."
    },
    {
        "sentence": "Vitamin D supplementation reduced the risk of experiencing at least one acute respiratory tract infection (adjusted odds ratio 0.88, 95% confidence interval 0.81 to 0.96; P for heterogeneity <0.001).",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "statistical_result",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Quantitative epidemiological result with odds ratio, confidence interval, and p-value."
    },
    {
        "sentence": "In patients with baseline 25-hydroxyvitamin D <25 nmol/L, vitamin D protected against ARI (adjusted odds ratio 0.30, 95% CI 0.17 to 0.53).",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "statistical_result",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Quantitative subgroup finding with exact effect size and confidence interval."
    },
    {
        "sentence": "We hypothesized that individual participant data meta-analysis would resolve discrepancies between previous aggregate data reviews.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "hypothesis",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Stated study hypothesis motivating the research design."
    },
    {
        "sentence": "Previous studies have reported contradictory findings regarding the efficacy of vitamin D in preventing respiratory tract infections.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "background",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Literature context describing prior inconsistent evidence."
    },
    {
        "sentence": "25-hydroxyvitamin D is the major circulating metabolite of vitamin D and the widely accepted marker of vitamin D status.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "background",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Biochemical domain fact and background definition."
    },
    {
        "sentence": "We searched Medline, Embase, and the Cochrane Central Register of Controlled Trials from inception to December 2015.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "methodology",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Literature search methodology and databases."
    },
    {
        "sentence": "Two investigators independently assessed risk of bias for each included study using the Cochrane Collaboration tool.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "methodology",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Quality assessment and study screening procedure."
    },
    {
        "sentence": "Data were obtained from 10,933 participants in 25 randomised controlled trials aged 0 to 95 years.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "dataset_description",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Cohort sample size and age distribution description."
    },
    {
        "sentence": "Our study has several limitations, including the observational nature of secondary outcome measures and heterogeneity in dosing regimens.",
        "label": 0,
        "paper_id": "Martineau et al. (2017)",
        "category": "limitation",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Explicit statement of study limitations and confounders."
    },
    {
        "sentence": "Our results support the administration of daily or weekly vitamin D to individuals with low baseline vitamin D status.",
        "label": 1,
        "paper_id": "Martineau et al. (2017)",
        "category": "ambiguous",
        "doi": "10.1136/bmj.i6583",
        "annotation_rationale": "Clinical conclusion and recommendation directly derived from the study findings."
    },

    # --- Paper 4: Bolland et al. (2022) [medRxiv 10.1101/2022.02.03.22270409] ---
    {
        "sentence": "After adjusting for cluster randomisation, vitamin D supplementation had no statistically significant effect on acute respiratory infection risk in the Blue Sky trial.",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "finding",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Direct reported finding from re-analysis of the trial."
    },
    {
        "sentence": "In each analysis, there is no statistically significant effect of vitamin D supplementation in any treatment group, or when the vitamin D groups are pooled and analysed together.",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "negative_finding",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Direct reported null finding across all trial arms."
    },
    {
        "sentence": "Our updated meta-analysis shows that vitamin D supplementation does not prevent acute respiratory infections in children.",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "negative_finding",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Empirical negative conclusion from updated meta-analysis."
    },
    {
        "sentence": "In individual patient data analyses, there was no statistically significant effect of vitamin D on ARI risk (odds ratio 0.58, 95% CI 0.26 to 1.29).",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "statistical_result",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Quantitative empirical result with odds ratio and confidence interval."
    },
    {
        "sentence": "Updated meta-analysis incorporating the corrected cluster-adjusted data yielded an odds ratio of 0.94 (95% CI 0.88 to 1.01), showing no significant protective effect.",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "statistical_result",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Quantitative updated meta-analysis result with confidence interval spanning unity."
    },
    {
        "sentence": "The results of both meta-analyses are heavily influenced by the Blue Sky study, a cluster-randomised trial of vitamin D in schoolchildren from Mongolia.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "background",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Background contextualization of prior literature."
    },
    {
        "sentence": "Cluster randomised trials are trials in which groups of individuals, rather than individuals themselves, are randomised to study interventions.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "background",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "General methodological definition in epidemiology."
    },
    {
        "sentence": "The recommended approach is to adjust the results of a cluster-randomised trial for the clustering of participants inherent in the design.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "methodology",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Statistical adjustment methodology rationale."
    },
    {
        "sentence": "We extracted data from the publication and replicated the published results: odds ratio 0.92 (0.86-0.99), I2= 36%.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "methodology",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Replication protocol and verification procedure."
    },
    {
        "sentence": "The Blue Sky trial included 1,430 schoolchildren from classrooms in six schools in Ulaanbaatar, Mongolia.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "dataset_description",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Trial cohort description and participant count."
    },
    {
        "sentence": "A limitation of our secondary analysis is that individual patient data from all trial arms were not publicly available.",
        "label": 0,
        "paper_id": "Bolland et al. (2022)",
        "category": "limitation",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Study limitation acknowledged by authors."
    },
    {
        "sentence": "These results dispute earlier claims that vitamin D produces substantial protective benefits in children with low baseline vitamin D status.",
        "label": 1,
        "paper_id": "Bolland et al. (2022)",
        "category": "ambiguous",
        "doi": "10.1101/2022.02.03.22270409",
        "annotation_rationale": "Contradictory claim asserting the study's challenge to previous literature."
    },

    # --- Paper 5: Jolliffe et al. (2024) [medRxiv 10.1101/2024.09.18.24313866] ---
    {
        "sentence": "In the subgroup aged 1.00 to 15.99 years receiving daily vitamin D dosing, supplementation showed a statistically significant protective effect against acute respiratory infection.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "finding",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Reported subgroup positive finding."
    },
    {
        "sentence": "For the primary comparison of any vitamin D vs. placebo, the intervention did not significantly affect overall ARI risk.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "negative_finding",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Primary null finding of the updated meta-analysis."
    },
    {
        "sentence": "Pre-specified subgroup analysis did not reveal evidence of effect modification by age, baseline vitamin D status, or dosing regimen.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "negative_finding",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Negative finding regarding effect modification across subgroups."
    },
    {
        "sentence": "Overall, vitamin D vs placebo did not significantly reduce ARI risk (OR 0.94, 95% CI 0.88-1.00, 40 studies, 61,589 participants).",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "statistical_result",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Statistical quantitative outcome for the overall primary outcome."
    },
    {
        "sentence": "In the subgroup aged 1.00 to 15.99 years, vitamin D showed a statistically significant protective effect (OR 0.74, 95% CI 0.60 to 0.92, 16 studies).",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "statistical_result",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Statistical quantitative result for pediatric subgroup."
    },
    {
        "sentence": "We hypothesized that the addition of data from recently completed large trials would clarify whether vitamin D reduces the risk of acute respiratory infection.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "hypothesis",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Stated study hypothesis motivating the updated review."
    },
    {
        "sentence": "A 2021 meta-analysis of 37 randomised controlled trials revealed a statistically significant protective effect of vitamin D supplementation against acute respiratory infections.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "background",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Citation of prior 2021 meta-analysis in introduction."
    },
    {
        "sentence": "We searched MEDLINE, EMBASE, the Cochrane Central Register of Controlled Trials, and ClinicalTrials.gov between May 2020 and April 2024.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "methodology",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Database search strategy and date window."
    },
    {
        "sentence": "Study quality and risk of bias were evaluated using the Cochrane risk-of-bias tool for randomized trials.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "methodology",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Standard methodology tool used for quality assessment."
    },
    {
        "sentence": "Data were combined with data from 48,488 participants in 43 previously identified RCTs, bringing the total to 64,086 participants in 46 trials.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "dataset_description",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Aggregate dataset sample size and study count breakdown."
    },
    {
        "sentence": "A major limitation is that individual participant data were not available for the newly identified trials.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "limitation",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Explicit limitation of aggregate-data approach."
    },
    {
        "sentence": "The 95% CI for this effect now spans 1.00, indicating no statistically significant protection.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "ambiguous",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Core takeaway sentence stating the lack of statistical significance."
    },
    {
        "sentence": "Sensitivity analyses excluding trials at high risk of bias yielded consistent point estimates with the primary analysis.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2024)",
        "category": "ambiguous",
        "doi": "10.1101/2024.09.18.24313866",
        "annotation_rationale": "Robustness check reporting methodological consistency rather than primary claim."
    },

    # --- Paper 6: Jolliffe et al. (2022) CORONAVIT Phase 3 RCT [medRxiv 10.1101/2022.03.22.22271707] ---
    {
        "sentence": "Among adults with a high baseline prevalence of sub-optimal vitamin D status, implementation of a population-level test-and-treat approach to vitamin D replacement did not reduce risk of all-cause ARI or COVID-19.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "negative_finding",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Main clinical conclusion and primary null finding of the Phase 3 trial."
    },
    {
        "sentence": "No statistically significant effect of either dose was seen on the primary outcome of incident doctor- or swab test-confirmed ARI, or on the major secondary outcome of incident swab test-confirmed COVID-19.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "negative_finding",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Direct reported null outcome on primary and secondary endpoints."
    },
    {
        "sentence": "Oral vitamin D supplementation was safe and well-tolerated at both doses investigated.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "finding",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Safety and tolerability outcome finding."
    },
    {
        "sentence": "Higher-dose vs no offer yielded an odds ratio of 1.09 (95% CI 0.82-1.46) for at least one ARI of any cause.",
        "label": 1,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "statistical_result",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Quantitative odds ratio and confidence interval on clinical outcome."
    },
    {
        "sentence": "We aimed to determine whether population-level implementation of a test-and-treat approach to correction of sub-optimal vitamin D status influences risk of all-cause acute respiratory infection.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "hypothesis",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Study aim and trial objective."
    },
    {
        "sentence": "The study was designed as a Phase 3 open-label randomised controlled trial.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "methodology",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Trial design classification."
    },
    {
        "sentence": "The trial enrolled 6200 adults aged 16 years or older who were not already taking vitamin D supplements at baseline.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "dataset_description",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Enrolled sample size and age eligibility criteria."
    },
    {
        "sentence": "This contrasts with recent findings from a Phase 2 RCT conducted in Mexican health care workers, which reported a strong protective effect of a daily dose of 4,000 IU vitamin D.",
        "label": 0,
        "paper_id": "Jolliffe et al. (2022) CORONAVIT",
        "category": "ambiguous",
        "doi": "10.1101/2022.03.22.22271707",
        "annotation_rationale": "Discussion section citing external trial results."
    },

    # --- Paper 7: Zhang et al. (2022) Treatment Meta-analysis [PMCID: PMC8955452] ---
    {
        "sentence": "In the meta-analysis of all the trials, vitamin D supplements had a beneficial effect in the treatment of ARIs (relative risk (RR) = 1.07; 95% confidence interval (CI), 1.01-1.13; I2 = 66.9%).",
        "label": 1,
        "paper_id": "Zhang et al. (2022)",
        "category": "statistical_result",
        "doi": "PMID: 35334804",
        "annotation_rationale": "Quantitative therapeutic meta-analysis outcome with relative risk and confidence interval."
    },
    {
        "sentence": "Recent randomized controlled trials have reported inconsistent findings regarding the efficacy of vitamin D supplementation in the treatment of acute respiratory infections.",
        "label": 0,
        "paper_id": "Zhang et al. (2022)",
        "category": "background",
        "doi": "PMID: 35334804",
        "annotation_rationale": "Background on conflicting prior trials in treatment."
    },
    {
        "sentence": "PubMed, EMBASE, and the Cochrane Library were searched for relevant articles in June 2021.",
        "label": 0,
        "paper_id": "Zhang et al. (2022)",
        "category": "methodology",
        "doi": "PMID: 35334804",
        "annotation_rationale": "Literature database search strategy."
    },
    {
        "sentence": "Out of 390 articles retrieved from the databases, we included 18 RCTs, which involved 3648 participants.",
        "label": 0,
        "paper_id": "Zhang et al. (2022)",
        "category": "dataset_description",
        "doi": "PMID: 35334804",
        "annotation_rationale": "Flow of studies and total participant sample size."
    }
]

# Write real evaluation dataset to tests/data/claim_extractor_eval.csv
eval_csv_path = ROOT / "tests" / "data" / "claim_extractor_eval.csv"
fieldnames = ["sentence", "label", "paper_id", "category", "doi", "annotation_rationale"]

with eval_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in REAL_PAPER_ROWS:
        writer.writerow(row)

print(f"Successfully wrote {len(REAL_PAPER_ROWS)} real-paper rows to {eval_csv_path}")

# Write separate synthetic evaluation dataset to tests/data/claim_extractor_synthetic_eval.csv
# This is explicitly separated from the real-paper evaluation dataset per user instructions.
SYNTHETIC_ROWS = [
    {
        "sentence": "Our synthetic model achieves a 99.5% accuracy on benchmark test suite Alpha.",
        "label": 1,
        "paper_id": "SYNTHETIC_TEST_FIXTURE_1",
        "category": "finding",
        "is_synthetic": True,
        "annotation_rationale": "Synthetic positive claim generated solely for smoke/unit testing."
    },
    {
        "sentence": "Synthetic evaluation reveals that treatment X produced a statistically significant reduction in synthetic metric Y.",
        "label": 1,
        "paper_id": "SYNTHETIC_TEST_FIXTURE_1",
        "category": "statistical_result",
        "is_synthetic": True,
        "annotation_rationale": "Synthetic positive finding generated solely for smoke/unit testing."
    },
    {
        "sentence": "We trained the synthetic neural network using stochastic gradient descent for 10 epochs.",
        "label": 0,
        "paper_id": "SYNTHETIC_TEST_FIXTURE_2",
        "category": "methodology",
        "is_synthetic": True,
        "annotation_rationale": "Synthetic method sentence generated solely for smoke/unit testing."
    },
    {
        "sentence": "The synthetic corpus consists of 500 artificial documents generated with random seed 42.",
        "label": 0,
        "paper_id": "SYNTHETIC_TEST_FIXTURE_2",
        "category": "dataset_description",
        "is_synthetic": True,
        "annotation_rationale": "Synthetic dataset description generated solely for smoke/unit testing."
    }
]

synthetic_csv_path = ROOT / "tests" / "data" / "claim_extractor_synthetic_eval.csv"
synth_fieldnames = ["sentence", "label", "paper_id", "category", "is_synthetic", "annotation_rationale"]

with synthetic_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=synth_fieldnames)
    writer.writeheader()
    for row in SYNTHETIC_ROWS:
        writer.writerow(row)

print(f"Successfully wrote {len(SYNTHETIC_ROWS)} synthetic rows to {synthetic_csv_path}")
