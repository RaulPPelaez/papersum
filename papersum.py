from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import transformers
import torch
import os
model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config,
    trust_remote_code=True,
    device_map="auto",
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

prompts = [
"Can you give me a very clear explanation of the core assertions, implications, and mechanics elucidated in this paper? Like you're talking to a CEO. So what? What's the bottom line here? Be as concise as posible, I must read it in under 30 seconds. Start your answer by stating the title of the paper. Below are the contents of the paper:",
]
#For every .txt file in the directory "input", run the model with the prompt and the text in the file, and save the output in the directory output.txt
for prompt in prompts:
    #Take on files ending in .txt
    for input_file in os.listdir("input"):
        if not input_file.endswith(".txt"):
            continue
        with open("input/" + input_file, "r") as f:
            text = f.read()
            sequences = pipeline(
                prompt + "\n-----------------\n" + text[:6000] + "\n-----------------\n",
                max_length=20000,
                do_sample=True,
                top_k=None,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            for seq in sequences:
                print(f"This is the output for {input_file}:")
                print(f"Result: {seq['generated_text']}")
                with open("output.txt", "a") as f:
                    f.write(seq['generated_text'])
