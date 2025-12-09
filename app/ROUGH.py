from rouge_score import rouge_scorer

def evaluate_summary(reference_summary, generated_summary):
    """
    Evaluate summarisation quality using ROUGE metrics.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {
        #defines a function to evaluate summarization quality using ROUGE metrics
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3)
    }
reference = """The team approved the project budget and discussed next steps."""
generated = """The project budget was approved, and future actions were planned."""


scores = evaluate_summary(reference, generated)
print("Summarisation Quality (ROUGE):", scores)

#Calculates ROUHG Metrics for summarization quality