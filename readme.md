# Emergence in Large Language Models: Testing the Scaling Hypothesis

## Overview
This project investigates whether large language models exhibit emergent abilities as they scale, testing the competing hypotheses of Wei et al. (2022) vs. Schaeffer et al. (2023) on BIG-Bench Hard tasks. The study systematically evaluates three prompt engineering techniques across three orders of magnitude in model scale.

## Research Questions
1. Do LLMs exhibit emergent reasoning abilities as they scale, or do capabilities improve smoothly?
2. Which prompting technique (zero-shot, few-shot, or chain-of-thought) is most effective across different model scales?
3. Is there a minimum viable scale for advanced prompting techniques?

## Key Findings
- **No evidence of sharp emergence** detected across 450 experiments
- Performance scales **smoothly and continuously** from 8B to 70B to Gemini parameters  
- Results support **Schaeffer et al.'s "mirage" hypothesis**
- Overall accuracy: **67.3%** on BIG-Bench Hard tasks
- **"Reasoning Gap" identified**: Chain-of-Thought exhibits negative transfer on 8B models (38.0% vs 48.0% zero-shot)
- Best performance: **Few-shot with Gemini 2.5 Pro (92%)**

## Methodology

### Models
- **Small**: Llama 3.1 8B (via Groq API, model: `llama3-8b-8192`)
- **Medium**: Llama 3.3 70B (via Groq API, model: `llama3-70b-8192`)  
- **Large**: Gemini 2.5 Pro (via Google AI Studio)

### Prompting Techniques
- **Zero-shot**: Direct question answering without examples
- **Few-shot**: Three demonstration examples (k=3) for in-context learning
- **Chain-of-Thought**: Step-by-step reasoning using BBH official prompts

### Tasks (BIG-Bench Hard)
1. `boolean_expressions` - Logical evaluation
2. `date_understanding` - Temporal reasoning
3. `geometric_shapes` - Spatial reasoning
4. `tracking_shuffled_objects_five_objects` - Object tracking
5. `word_sorting` - Symbolic manipulation

### Experimental Design
- Total experiments: 450 (3 models × 3 techniques × 5 tasks × 10 examples)
- Temperature: 0.0 (deterministic)
- Random seed: 42
- Statistical significance: Chi-square tests with α=0.05

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Groq API key (for Llama models)
- Google AI Studio API key (for Gemini 2.5 Pro)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/rahulmandadi/emergence-scaling-project.git
cd emergence-scaling-project
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**
```bash
cp config.template.yaml config.yaml
# Edit config.yaml to add your API keys
```

5. **Download BIG-Bench Hard dataset**
```bash
cd data/bbh
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git .
cd ../..
```

## Usage

### Data Collection
```bash
# Run complete experiment (approximately 2-3 hours due to rate limits)
python src/data_collector.py

# The system automatically resumes from checkpoint if interrupted
```

### Analysis
```bash
# Generate statistical analysis and visualizations
python src/analyzer.py
```

### Utility Scripts
```bash
# Clean CoT experiments for re-running
python scripts/remove_cot.py

# Verify task file integrity
python verify_tasks.py
```

## Project Structure
```
emergence-scaling-project/
├── src/
│   ├── data_collector.py      # Main experiment orchestrator with checkpointing
│   ├── analyzer.py            # Statistical analysis and visualization pipeline
│   ├── api_clients.py         # Groq and Google API interfaces
│   ├── prompt_builder.py      # Dynamic prompt construction with BBH CoT support
│   └── response_parser.py     # Multi-format response extraction with regex cascade
├── data/
│   ├── bbh/                   
│   │   ├── bbh/               # BIG-Bench Hard task JSON files
│   │   └── cot-prompts/       # Official Chain-of-Thought demonstrations
│   ├── processed/             # Experimental results (CSV format)
│   └── checkpoints/           # Resume capability for interrupted runs
├── results/
│   └── figures/               # Generated plots and summary statistics
├── scripts/
│   └── remove_cot.py         # Data cleanup utility
├── config.yaml               # API keys and hyperparameters (not in repository)
├── config.template.yaml      # Configuration template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Results

### Performance Summary

| Model | Zero-shot | Few-shot | Chain-of-Thought |
|-------|-----------|----------|------------------|
| **Llama 3.1 8B** | 48.0% | 44.0% | 38.0% |
| **Llama 3.3 70B** | 62.0% | 86.0% | 74.0% |
| **Gemini 2.5 Pro** | 76.0% | 92.0% | 86.0% |

### Key Statistics
- **Overall Accuracy**: 67.3%
- **Best Technique**: Few-shot (74.0% average across models)
- **Best Model**: Gemini 2.5 Pro (84.7% average across techniques)
- **Largest Scaling Jump**: 8B to 70B for few-shot (+42 percentage points)
- **Statistical Significance**: Achieved for Llama 70B few-shot vs zero-shot (p=0.0121)
- **Parser Success Rate**: 88% (12% empty predictions due to response format variations)

### Task Performance Ranking
1. `boolean_expressions`: 90.0%
2. `tracking_shuffled_objects_five_objects`: 80.0%
3. `date_understanding`: 68.9%
4. `geometric_shapes`: 64.4%
5. `word_sorting`: 33.3%

### Cost-Benefit Analysis
- **Llama 8B**: CoT requires 748 tokens per correct answer vs 126 for zero-shot (5.9x more expensive)
- **Llama 70B**: CoT efficiency improves to 384 tokens per correct answer
- **Gemini 2.5 Pro**: CoT becomes viable at 331 tokens per correct answer

## Academic Context

- **Course**: DS 5983 - Large Language Models
- **Institution**: Northeastern University
- **Semester**: Fall 2024
- **Project Type**: Final Research Project

## Key Contributions

1. **Empirical adjudication** of the emergence debate using rigorous experimental design
2. **Identification of the "Reasoning Gap"**: Discovery that CoT exhibits negative transfer on sub-10B models
3. **Cost-efficiency analysis**: Quantification of token costs per correct answer across techniques
4. **Reproducible pipeline**: Checkpoint-based system for resumable experiments

## Limitations

1. **Parser constraints**: 12% of responses failed extraction, particularly affecting the word_sorting task
2. **Limited task coverage**: 5 of 23 BBH tasks evaluated due to computational constraints
3. **Model selection**: Restricted to freely available API tiers
4. **Sample size**: 10 examples per task-model-technique combination

## References

1. Wei, J., et al. (2022). "Emergent Abilities of Large Language Models." *Transactions on Machine Learning Research*.
2. Schaeffer, R., et al. (2023). "Are Emergent Abilities of Large Language Models a Mirage?" *arXiv preprint arXiv:2304.15004*.
3. Suzgun, M., et al. (2022). "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them." *arXiv preprint arXiv:2210.09261*.
4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*.
5. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." *arXiv preprint arXiv:2001.08361*.

## Citation

```bibtex
@misc{mandadi2025emergence,
  author = {Mandadi, Rahul Reddy},
  title = {Emergent or Gradual? Scaling Behavior of Prompt Engineering Techniques},
  year = {2025},
  institution = {Northeastern University},
  course = {DS 5983 - Large Language Models},
  url = {https://github.com/rahulmandadi/emergence-scaling-project}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- BIG-Bench Hard dataset and Chain-of-Thought prompts from [Suzgun et al. (2022)](https://github.com/suzgunmirac/BIG-Bench-Hard)
- [Groq](https://groq.com) for Llama model API access
- [Google AI Studio](https://aistudio.google.com) for Gemini 2.5 Pro access
- Course instructor and teaching assistants for guidance

## Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---
*This project was completed as part of the requirements for DS 5983: Large Language Models at Northeastern University, Fall 2025.*