import json, os
files = {
    "baseline": "results/metrics.json",
    "pa_word": "results/metrics_pa_word.json",
    "char_svm": "results/metrics_char_svm.json",  # if exists
}
rows = []
for name, path in files.items():
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append([name, m.get("accuracy"), m.get("precision"), m.get("recall"), m.get("f1"), m.get("n_test")])
if rows:
    print("model,accuracy,precision,recall,f1,n_test")
    for r in rows:
        print(",".join(str(x) for x in r))
else:
    print("No metrics found.")
