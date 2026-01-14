import re
import numpy as np
import hashlib

from dataclasses import dataclass
from corewar import redcode, Warrior

from llm_async import GPT

@dataclass
class GPTWarrior:
    prompt: str
    llm_response: str
    warrior: Warrior = None
    error: str = None
    id: str = None
    parent_id: str = None

    full_outputs: dict = None
    outputs: dict = None
    fitness: float = -np.inf
    bc: tuple[int, int] | None = None


class CorewarGPT():
    def __init__(self, model, system_prompt, new_warrior_prompt, mutate_warrior_prompt, temperature=1., environment=None, base_url=None):
        self.new_warrior_prompt = new_warrior_prompt
        self.mutate_warrior_prompt = mutate_warrior_prompt
        self.gpt = GPT(model=model, system_prompt=system_prompt, temperature=temperature, base_url=base_url)
        self.environment = environment
        self.all_generations = [] # list of tuples of (generation_type, gpt_warriors)

    def parse_llm_response(self, prompt, llm_response):
        gpt_warrior = GPTWarrior(prompt=prompt, llm_response=llm_response)
        try:
            # a = re.search(r"```.*\n[\s\S]*?```", warrior_str)
            llm_response = re.sub(r"```.*", "", llm_response) # remove the backticks and language tag
            warrior = redcode.parse(llm_response.split("\n"), self.environment)
            gpt_warrior.warrior = warrior
        except Exception as e:
            gpt_warrior.error = str(e)
        gpt_warrior.id = hashlib.sha256(gpt_warrior.llm_response.encode()).hexdigest()
        return gpt_warrior

    async def new_warrior_async(self, n_warriors=1, n_responses=1):
        prompts = [self.new_warrior_prompt for _ in range(n_warriors)]
        llm_responses = await self.gpt.get_multiple_completions_async(prompts, n_responses=n_responses) # list of list of strs
        gpt_warriors = [[None for _ in range(n_responses)] for _ in range(n_warriors)]
        for i_warrior in range(n_warriors):
            for i_response in range(n_responses):
                gpt_warrior = self.parse_llm_response(prompts[i_warrior], llm_responses[i_warrior][i_response])
                gpt_warriors[i_warrior][i_response] = gpt_warrior
        self.all_generations.append(("new_warrior", gpt_warriors))
        return np.array(gpt_warriors, dtype=object)
        
    async def mutate_warrior_async(self, gpt_warriors, n_responses=1):
        old_gpt_warriors = gpt_warriors
        prompts = [f"{self.mutate_warrior_prompt}\n\n\n{gpt_warrior.llm_response}" for gpt_warrior in old_gpt_warriors]
        llm_responses = await self.gpt.get_multiple_completions_async(prompts, n_responses=n_responses) # list of list of strs
        gpt_warriors = [[None for _ in range(n_responses)] for _ in range(len(llm_responses))]
        for i_warrior in range(len(llm_responses)):
            for i_response in range(n_responses):
                gpt_warrior = self.parse_llm_response(prompts[i_warrior], llm_responses[i_warrior][i_response])
                gpt_warrior.parent_id = old_gpt_warriors[i_warrior].id
                gpt_warriors[i_warrior][i_response] = gpt_warrior
        self.all_generations.append(("mutate_warrior", gpt_warriors))
        return np.array(gpt_warriors, dtype=object)
