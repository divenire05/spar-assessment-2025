import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

prompts = [
    "Translate to pirate: Hello, friend!",
    "Translate to pirate: I love AI.",
    "Translate to pirate: Where is the treasure?"
]

# Load base and finetuned
base_pipe = pipeline("text-generation", model="distilgpt2")
finetuned_pipe = pipeline("text-generation", model="./finetuned_model")

results = []
for p in prompts:
    base_out = base_pipe(p, max_new_tokens=30, num_return_sequences=1)[0]["generated_text"]
    finetuned_out = finetuned_pipe(p, max_new_tokens=30, num_return_sequences=1)[0]["generated_text"]

    results.append({
        "prompt": p,
        "pretrained_output": base_out,
        "finetuned_output": finetuned_out
    })
    print(f"\nPrompt: {p}\nBase: {base_out}\nFinetuned: {finetuned_out}")

# Save results to JSON & Markdown
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("eval_results.md", "w") as f:
    f.write("| Prompt | Pretrained Output | Finetuned Output |\n")
    f.write("|--------|------------------|------------------|\n")
    for r in results:
        f.write(f"| {r['prompt']} | {r['pretrained_output']} | {r['finetuned_output']} |\n")

print("Results saved in eval_results.json and eval_results.md")