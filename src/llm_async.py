import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from openai import AsyncOpenAI
import openai
import backoff

# MAX_NUM_TOKENS = 4096

class GPT:
    def __init__(self, model, system_prompt, temperature=1., base_url=None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url)

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.PermissionDeniedError))
    async def get_completion_async(self, prompt, n_responses=1):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            # max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        return [r.message.content for r in response.choices]

    async def get_multiple_completions_async(self, prompts, n_responses=1):
        results = await asyncio.gather(*(self.get_completion_async(p, n_responses) for p in prompts))
        return results
    
    def get_completion(self, prompt, n_responses=1):
        return asyncio.run(self.get_completion_async(prompt, n_responses))

    def get_multiple_completions(self, prompts, n_responses=1):
        return asyncio.run(self.get_multiple_completions_async(prompts, n_responses))
        
if __name__ == "__main__":
    gpt = GPT(model="gpt-4o-mini", system_prompt="You are a helpful assistant.")
    prompts = ["Hello", "Tell me a joke", "What's 2+2?", "What's the capital of France?"]

    for p in prompts:
        c = gpt.get_completion(p)
        print(c)

    results = gpt.get_multiple_completions(prompts, n_responses=2)
    print(results)
    
