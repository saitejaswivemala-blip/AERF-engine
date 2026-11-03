from db.repository import Repository
from nlp.claim_extractor import ScientificClaimExtractor

repo = Repository("experiments/runs/experiment_FINAL_2026-09-02.db")
extractor = ScientificClaimExtractor(threshold=0.55)

chunks = repo.get_chunks_by_paper(9)[:5]
classifier = extractor._classifier_instance()

for i, chunk in enumerate(chunks, 1):
    doc = extractor._doc(chunk.text)
    sentences = [s for s in doc.sents if extractor._eligible(s)]
    print(f"\n--- CHUNK {i}: {len(sentences)} eligible sentences ---")

    if not sentences:
        print("No sentences passed the eligibility filter.")
        continue

    outputs = classifier(
        [s.text for s in sentences[:5]],
        list(extractor.LABELS),
        multi_label=False
    )

    if isinstance(outputs, dict):
        outputs = [outputs]

    for sentence, result in zip(sentences[:5], outputs):
        print("TEXT:", sentence.text[:180])
        print("PREDICTION:", result["labels"][0])
        print("SCORE:", round(float(result["scores"][0]), 3))
