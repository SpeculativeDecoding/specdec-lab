## Speculative Decoding Project Lab

We are exploring speculative decoding techniques and optimizations.

# Existing work: 
PRISM (MLSys’26), EAGLE-3 (NeurIPS’25), EAGLE (ICML’24), TALON(2026)

# Objective: 
Comparison of best-performing tree shape per workload

# Research question: 
Under a fixed candidate budget, can a tree strategy be chosen for a workload?

# Method: 
Evaluate existing works on SGLang as testing env. Eg. EAGLE-3 drafter EAGLE3-LLaMA3.1-Instruct-8B

# Metrics: 
Measure acceptance by draft step, accepted tokens per target pass, end-to-end tokens/s.

# Data: 
MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail and Natural Questions
