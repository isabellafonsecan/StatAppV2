from jiwer import process_words

out = process_words(
    "this is the ground truth",
    "this is the ground truth .",
)

print("WER:", out.wer)
print("Substitutions:", out.substitutions)
print("Deletions:", out.deletions)
print("Insertions:", out.insertions)
print("Hits (correct):", out.hits)