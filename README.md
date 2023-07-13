## Papersum
Summarize arxiv papers.
This tool will query arxiv for recent papers on a certain subject and will ask a large language model to summarize them.

You can use the OpenAI API for this (with any of its available models) or run any transformer from Huggingface locally on your machine (warning, this typically requires ~100GB of VRAM available)


### Installation

Run ```pip install -e .```

### Usage

Run ```papersum -h```

### Examples
Use the falcon pipeline for huggingface models (even non-falcon ones)
```bash
papersum --topic="neural network potentials" --max_papers=1 --pipeline=falcon --model="tiiuae/falcon-40b-instruct"
```
 
Use the openai pipeline for OpenAI models
```bash
papersum --topic="neural network potentials" --max_papers=1 --pipeline=openai --model="gpt-3.5-turbo-16k"
```


