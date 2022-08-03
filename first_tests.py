#%%
from math import exp
import os
import openai
from transformers import GPT2Tokenizer

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(tokenizer("\n")["input_ids"])
#%%
openai.api_key = OPENAI_API_KEY

# List Engines (Models)
engines = openai.Engine.list()
# Print all engines IDs
for engine in engines.data:
    print(engine.id)
#%%
prompt = """Rate these sentences offensiveness:
###
You suck ass.
Rating from 1 to 5: 5
###
You're not very nice
Rating from 1 to 5: 2
###
Your mother probably getting money while having orgasms.
Rating from 1 to 5: 5
###
I like you.
Rating from 1 to 5: 1
###
I hate you.
Rating from 1 to 5:
"""

# def generate_prompt(examples: list[str], title: str="")


def get_rank_probs(prompt: str, engine: str = "text-babbage-001"):
    unintersting_tokens = {"newline": "198", "EOS": "50256"}
    logit_bias = dict((v, -100) for v in unintersting_tokens.values())
    completion = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        logprobs=5,
        max_tokens=1,
        stream=False,
        logit_bias=logit_bias,
    )
    top_logprobs = completion["choices"][0]["logprobs"]["top_logprobs"][0]
    probs = [
        exp(top_logprobs[str(i)]) if str(i) in top_logprobs else 0 for i in range(1, 6)
    ]
    return probs


#%%


# Print each token as it is returned
print(get_rank_probs(prompt))

# %%
