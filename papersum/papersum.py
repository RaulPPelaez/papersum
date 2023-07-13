import os
import argparse
from halo import Halo
from .arxiv import fetch_arxiv_papers
from time import sleep
import sys

prompts = [
    "Can you give me a very clear explanation of the core assertions, implications, and mechanics elucidated in this paper? Like you're talking to a CEO. So what? What's the bottom line here? Be as concise as posible, I must read it in under 30 seconds. Start your answer by stating the title of the paper. Below are the contents of the paper:",
]

class HiddenPrints:
    def __enter__(self):
        # self._original_stdout = sys.stdout
        # self._original_stderr = sys.stderr
        # sys.stdout = open(os.devnull, 'w')
        # sys.stderr = open(os.devnull, 'w')
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        # sys.stdout.close()
        # sys.stdout = self._original_stdout
        # sys.stderr.close()
        # sys.stderr = self._original_stderr
        pass

def generate_prompt(prompt, text, max_chars):
    return "Customer:\n" + prompt + "\n-----------------\n"+ text[:max_chars]+ "\n-----------------\n" + "Assistant:\n"



class FalconPipeline:
    def __init__(self, model_id=None):
        with HiddenPrints():
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

        model_id = model_id if model_id is not None else "tiiuae/falcon-40b-instruct"
        print(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, truncation="longest_first",)
        self.max_tokens = self.tokenizer.model_max_length
        print(f"Max tokens: {self.max_tokens}")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config,
            trust_remote_code=True,
            device_map="auto",
        )
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def count_tokens(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def generate(self, prompt, paper, max_chars=None):
        if max_chars is None:
            max_chars = 3000 if self.max_tokens == 2048 else 16000
        prompt = generate_prompt(prompt, paper, max_chars)
        while self.count_tokens(prompt) > self.max_tokens-512:
             prompt = prompt[:int(len(prompt)*0.8)]
        #Ensure that the prompt is an str
        sequences = self.pipeline(
            prompt,
            max_length=self.max_tokens,
            do_sample=True,
            top_k=None,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return sequences[0]["generated_text"]


class OpenAIPipeline:
    def __init__(self, model_id=None, temperature=0.01, api_key_file="key_openai.txt"):
        import openai

        self.model_id = model_id if model_id is not None else "gpt-3.5-turbo-16k"
        self.max_tokens = 16384
        self.temperature = temperature
        if not os.path.isfile(api_key_file):
            raise ValueError(
                f"API key file {api_key_file} not found. Please write your API key to this file."
            )
        with open(api_key_file, 'r', encoding='utf-8', errors='ignore') as infile:
            openai.api_key = infile.read().strip()

    def count_tokens(self, text):
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def generate(self, prompt, paper, max_chars=16000):
        import openai
        retry = 0
        max_retry = 3
        while retry < max_retry:
            try:
                conversation = [{'role':'system', 'content': paper}]
                conversation.append({'role':'user', 'content': prompt})
                response = openai.ChatCompletion.create(
                    model=self.model_id, messages=conversation, temperature=self.temperature
                )
                text = response["choices"][0]["message"]["content"]
                return text
            except Exception as oops:
                print(f'\n\nError communicating with OpenAI: "{oops}"')
                if "maximum context length" in str(oops):
                    paper = paper[:int(len(paper) * 0.8)]
                    continue
                retry += 1
                if retry >= max_retry:
                    print(f"\n\nExiting due to excessive errors in API: {oops}")
                    exit(1)
                print(f"\n\nRetrying in {2 ** (retry - 1) * 5} seconds...")
                sleep(2 ** (retry - 1) * 5)


def run():
    parser = argparse.ArgumentParser(description="This is a tool to summarize papers.")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="falcon",
        help='The pipeline to use. Can be "openai" or "falcon".',
    )
    parser.add_argument(
        "--prompt", type=str, default=prompts[0], help="The prompt to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use. Can be any gpt model when using the OpenAI pipeline, or any huggingface model when using the falcon pipeline.",
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        default="key_openai.txt",
        help="The file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="neural network potentials",
        help="The topic to search for in arxiv",
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=4,
        help="The maximum number of papers to summarize.",
    )
    args = parser.parse_args()
    if args.pipeline == "openai":
        pipeline = OpenAIPipeline(api_key_file=args.api_key_file, model_id=args.model)
    elif args.pipeline == "falcon":
        pipeline = FalconPipeline(model_id=args.model)
    else:
        raise ValueError("The pipeline must be either 'openai' or 'falcon'.")
    prompt = args.prompt
    papers = fetch_arxiv_papers(topic=args.topic, maxpapers=args.max_papers)
    # Take on files ending in .txt
    for title, text in papers.items():
        print(f"Summarizing article {title}:")
        ntokens = pipeline.count_tokens(text)
        print(f"Article contains {ntokens} tokens")
        if ntokens > pipeline.max_tokens:
            print(
                f"Warning, maximum tokens for model surpassed, paper text will be truncated to fit context window"
            )
        spinner = Halo(text="Thinking...", spinner="dots")
        spinner.start()
        response = pipeline.generate(prompt, text)
        spinner.stop()
        print(f"Result: {response}")
        with open("output.txt", "a") as f:
            f.write(prompt + "\n" + response + "\n")
