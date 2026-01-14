
import re
import random
import os
import numpy as np
import time
import hashlib
import os
import psutil
import copy

from dataclasses import dataclass, field
import tyro
import asyncio
from tqdm.auto import tqdm
from collections import defaultdict

from llm_corewar import CorewarGPT, GPTWarrior

from corewar_util import SimulationArgs, simargs_to_environment, parse_warrior_from_file, run_multiple_rounds
from corewar import MARS, Warrior
import util

@dataclass
class Args:
    # General arguments
    seed: int = 0
    save_dir: str | None = None
    n_processes: int = 24
    resume: bool | None = False # resume training from save_dir if it exists
    job_timeout: int = 24 * 60 * 60 # entire job timeout in seconds

    # Core War arguments
    simargs: SimulationArgs = field(default_factory=SimulationArgs) # Simulation arguments
    timeout: int = 900 # timeout for each simulation in seconds

    # DRQ arguments
    initial_opps: list[str] = field(default_factory=list) # list of initial opponents
    n_rounds: int = 10 # number of ruonds of DRQ
    n_iters: int = 100 # iterations of evolution per round
    log_every: int = 10 # log every n iterations
    last_k_opps: int | None = None # number of previous rounds' champions to use for current round
    sample_new_percent: float = 0.1 # probability of sampling a new warrior from LLM
    bc_axes: str = "tsp,mc" # comma separated list of two bc axes to use
    # crossover_prob: float = 0.0 # probability of crossover
    warmup_with_init_opps: bool | None = False
    warmup_with_past_champs: bool | None = False
    n_init: int = 8 # number of initial warriors
    n_mutate: int = 1 # number of mutated warriors
    fitness_threshold: float = 0.8 # if this fitness is not reached, continue to next round
    single_cell: bool | None = False # for testing: only use one cell in map elites

    # LLM arguments
    gpt_model: str = "gpt-4.1-mini-2025-04-14" # The GPT model to use
    api_base_url: str | None = None # Base URL for OpenAI-compatible APIs (e.g. https://openrouter.ai/api/v1)
    temperature: float = 1.0
    system_prompt: str = os.path.expanduser("./prompts/system_prompt_0.txt")
    new_prompt: str = os.path.expanduser("./prompts/new_prompt_0.txt")
    mutate_prompt: str = os.path.expanduser("./prompts/mutate_prompt_0.txt")

class MapElites:
    def __init__(self):
        self.archive = {} # bc -> phenotype
        self.history = [] # list of phenotypes which were placed

        self.coverage_history = [] # history of coverage at every place step
        self.fitness_history = [] # history of best fitness in the archive at every place step

    def sample(self):
        random_key = random.choice(list(self.archive.keys()))
        return self.archive[random_key]

    def place(self, phenotype):
        place = (phenotype.bc is not None) and (phenotype.fitness is not None)
        place = place and ((phenotype.bc not in self.archive) or (phenotype.fitness > self.archive[phenotype.bc].fitness))
        if place:
            self.archive[phenotype.bc] = phenotype
        self.history.append(phenotype)
        self.coverage_history.append(len(self.archive))
        self.fitness_history.append(self.get_best().fitness if len(self.archive) > 0 else -np.inf)
        return place
    
    def get_best(self):
        best_key, best_fitness = None, -np.inf
        for k, v in self.archive.items():
            if v.fitness > best_fitness:
                best_key, best_fitness = k, v.fitness
        return self.archive[best_key] if best_key in self.archive else None

class Main:
    """
    This is the main class to run DRQ.
    """
    def __init__(self, args: Args):
        self.args = args
        print(args)
        nproc = os.popen("nproc").read().strip()
        nproc_all = os.popen("nproc --all").read().strip()
        print(f"Number of cores: {nproc} / {nproc_all}")

        random.seed(args.seed)
        np.random.seed(args.seed)

        with open(args.system_prompt, "r") as f:
            system_prompt = f.read()
        with open(args.new_prompt, "r") as f:
            new_warrior_prompt = f.read()
        with open(args.mutate_prompt, "r") as f:
            mutate_warrior_prompt = f.read()
        # with open(args.crossover_prompt, "r") as f:
            # task_crossover_warrior = f.read()

        self.corewar_gpt = CorewarGPT(args.gpt_model, system_prompt, new_warrior_prompt, mutate_warrior_prompt,
                                      temperature=args.temperature, environment=simargs_to_environment(args.simargs),
                                      base_url=args.api_base_url)

        self.init_opps = []
        for file in args.initial_opps:
            warrior_str, warrior = parse_warrior_from_file(args.simargs, file)
            gpt_warrior = GPTWarrior(prompt=file, llm_response=warrior_str, warrior=warrior)
            gpt_warrior.id = hashlib.sha256(gpt_warrior.llm_response.encode()).hexdigest()
            self.init_opps.append(gpt_warrior)
        print(f"Loaded {len(self.init_opps)} opponent warriors")

        self.timestamps = []
        self.all_rounds_map_elites = {i_round: MapElites() for i_round in range(self.args.n_rounds)} # map elites of each round
    
    def get_fitness(self, phenotype):
        return phenotype.outputs["score"].item()

    def get_bc_features(self, phenotype):
        if self.args.single_cell:
            return (0, 0)
        if phenotype.outputs is None:
            return (0, 0)
        tsp, mc = phenotype.outputs['total_spawned_procs'].item(), phenotype.outputs['memory_coverage'].item()
        unique_opcodes = len({i.opcode for i in phenotype.warrior.instructions})
        program_len = len(phenotype.warrior.instructions)

        for bc, a in enumerate([1, 10, 100, 1000, 10000, np.inf]):
            if tsp < a:
                bc_tsp = bc
                break
        
        for bc, a in enumerate([10, 100, 500, 1000, 4000, np.inf]):
            if mc < a:
                bc_mc = bc
                break

        for bc, a in enumerate([4, 6, 8, 10, 14, np.inf]):
            if unique_opcodes < a:
                bc_uo = bc
                break

        for bc, a in enumerate([5, 12, 20, 35, 60, np.inf]):
            if program_len < a:
                bc_pl = bc
                break
        all_bcs = dict(tsp=bc_tsp, mc=bc_mc, uo=bc_uo, pl=bc_pl)
        bc1, bc2 = self.args.bc_axes.split(",")
        bc1 = all_bcs[bc1]
        bc2 = all_bcs[bc2]
        # print(f"bc1: {bc1}, bc2: {bc2}")
        return (bc1, bc2)

    def process_warrior(self, i_round, gpt_warrior):
        gpt_warrior = copy.deepcopy(gpt_warrior)
        map_elites = self.all_rounds_map_elites[i_round]
        # print(f"Processing with {list(range(i_round))} prev champs")
        prev_champs = [self.all_rounds_map_elites[i].get_best() for i in range(i_round)]
        # print([p is None for p in prev_champs])
        prev_champs = prev_champs[-self.args.last_k_opps:] if self.args.last_k_opps is not None else prev_champs
        # print("processing with previous champions:")
        # print([f"{pc.warrior.name}, {pc.fitness}" for pc in prev_champs])

        if gpt_warrior.warrior is None:
            gpt_warrior.bc, gpt_warrior.fitness = None, -np.inf
        else:
            opps = self.init_opps + prev_champs
            warriors = [w.warrior for w in [gpt_warrior, *opps]]
            outputs = run_multiple_rounds(self.args.simargs, warriors, n_processes=self.args.n_processes, timeout=self.args.timeout)
            if outputs is None:
                gpt_warrior.bc, gpt_warrior.fitness = None, -np.inf
            else:
                gpt_warrior.outputs = {k: v.mean(axis=-1)[0] for k, v in outputs.items()}
                gpt_warrior.fitness = self.get_fitness(gpt_warrior)
                # gpt_warrior.fitness = self.get_fitness(gpt_warrior) * len(warriors) # normalize the fitness by the number of opponents
                gpt_warrior.bc = self.get_bc_features(gpt_warrior)
                # print(f"Processed Warrrior {gpt_warrior.warrior.name} with fitness {gpt_warrior.fitness} and bc {gpt_warrior.bc}")
        map_elites.place(gpt_warrior)

    def init_round(self, i_round):
        initial_gpt_warriors = asyncio.run(self.corewar_gpt.new_warrior_async(n_warriors=1, n_responses=self.args.n_init)).flatten()
        for w in initial_gpt_warriors:
            self.process_warrior(i_round, w)

        if self.args.warmup_with_init_opps:
            for w in self.init_opps:
                self.process_warrior(i_round, w)
        if self.args.warmup_with_past_champs:
            prev_champs = [self.all_rounds_map_elites[i].get_best() for i in range(i_round)]
            prev_champs = prev_champs[-self.args.last_k_opps:] if self.args.last_k_opps is not None else prev_champs
            for w in prev_champs:
                self.process_warrior(i_round, w)

    def step(self, i_round):
        if random.random() < self.args.sample_new_percent or len(self.all_rounds_map_elites[i_round].archive) == 0:
            gpt_warriors = asyncio.run(self.corewar_gpt.new_warrior_async(n_warriors=1, n_responses=self.args.n_mutate)).flatten()
            for w in gpt_warriors:
                self.process_warrior(i_round, w)
        else:
            gpt_warrior = self.all_rounds_map_elites[i_round].sample()
            gpt_warriors_mutated = asyncio.run(self.corewar_gpt.mutate_warrior_async([gpt_warrior], n_responses=self.args.n_mutate)).flatten()
            for w in gpt_warriors_mutated:
                self.process_warrior(i_round, w)
    
    def run(self):
        this_job_start_time = time.time()
        if self.args.resume and os.path.exists(f"{self.args.save_dir}/args.pkl"):
            self.timestamps = util.load_pkl(self.args.save_dir, "timestamps")
            self.all_rounds_map_elites = util.load_pkl(self.args.save_dir, "all_rounds_map_elites")
            self.corewar_gpt.all_generations = util.load_pkl(self.args.save_dir, "all_generations")
            print(f"Resumed training from {self.args.save_dir}")

            start_abs_iter = self.timestamps[-1]["abs_iter"] + 1 # start from the next iteration
        else:
            start_abs_iter = 0

        pbar = tqdm(range(start_abs_iter, self.args.n_rounds * self.args.n_iters))
        for abs_iter in pbar:
            i_round, i_iter = abs_iter // self.args.n_iters, abs_iter % self.args.n_iters
            # print(f"Starting {abs_iter=}, {i_round=}, {i_iter=}")
            start_time = time.time()

            me = self.all_rounds_map_elites[i_round]
            best_fitness = me.get_best().fitness if len(me.archive) > 0 else -np.inf
            should_skip = best_fitness > self.args.fitness_threshold

            if not should_skip:
                if i_iter == 0:
                    self.init_round(i_round)
                self.step(i_round)

            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss  # Resident Set Size: memory used in bytes
            vms = process.memory_info().vms  # Virtual Memory Size: total virtual memory used in bytes
            self.timestamps.append(dict(abs_iter=abs_iter, i_round=i_round, i_iter=i_iter, dt=time.time()-start_time, rss=rss, vms=vms))

            if abs_iter % self.args.log_every == 0:
                self.save()
            
            if len(me.archive) > 0:
                pbar.set_postfix(best_fitness=me.get_best().fitness)

            if (time.time() - this_job_start_time) > self.args.job_timeout:
                break # manual stop after to avoid slurm job timeout
        self.save()
    
    def save(self):
        if self.args.save_dir is not None:
            util.save_pkl(self.args.save_dir, "args", self.args)
            util.save_pkl(self.args.save_dir, "timestamps", self.timestamps)
            util.save_pkl(self.args.save_dir, "all_rounds_map_elites", self.all_rounds_map_elites)
            util.save_pkl(self.args.save_dir, "all_generations", self.corewar_gpt.all_generations)
            for i_round, me in self.all_rounds_map_elites.items():
                if len(me.archive) > 0:
                    champion = me.get_best()
                    code = re.sub(r"```.*", "", champion.llm_response) # remove the backticks and language tag
                    with open(f"{self.args.save_dir}/round_{i_round:03d}_champion.red", "w") as f:
                        f.write(code)

if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.run()
    