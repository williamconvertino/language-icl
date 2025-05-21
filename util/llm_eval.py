import os
from .device import get_device
from openai import OpenAI
from dotenv import load_dotenv
from .evaluator import generate_text_nucleus
import statistics
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), "../llm_eval")
LLM_MODEL = 'gpt-4o'

SYSTEM_PROMPT = "You are a writing evaluator designed to assess student story completions. You will be provided children's stories written for a 3-4 year old audience. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

USER_PROMPT = """
In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

Here is the pre-written beginning:

<PROVIDED BEGINNING>
[STORY_BEGIN]
</PROVIDED BEGINNING>

Here is the students response:

<STUDENT RESPONSE>
[STORY_END]
</STUDENT RESPONSE>

First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

1. Grammar:
* Is the writing grammatically correct?
* Evaluate syntax, punctuation, and sentence structure.
2. Consistency:
* Is the student's writing consistent with the provided beginning of the story?
* How well does the student complete the final sentence of the prescribed beginning?
3. Plot:
* Does the plot of the student's writing make sense (regardless of the provided beginning)?
4. Creativity: 
* How creative is the student's writing?

Format your response as follows:

<GRAMMAR>
[Qualitative assessment of grammar]
</GRAMMAR>
<GRAMMAR_GRADE>
[Grade out of 10]
</GRAMMAR_GRADE>

<CONSISTENCY>
[Qualitative assessment of consistency]
</CONSISTENCY>
<CONSISTENCY_GRADE>
[Grade out of 10]
</CONSISTENCY_GRADE>

<PLOT>
[Qualitative assessment of plot]
</PLOT>
<PLOT_GRADE>
[Grade out of 10]
</PLOT_GRADE>

<CREATIVITY>
[Qualitative assessment of creativity]
</CREATIVITY>
<CREATIVITY_GRADE>
[Grade out of 10]
</CREATIVITY_GRADE>

Provide your assessment below:
"""

class LLMEvaluator:
    def __init__(self, model, tokenizer, splits, checkpoint=None):
        self.model = model
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.test_loader = splits["test"]
        self.device = get_device()
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "OPENAI_API_KEY is not set. Please set it in your environment variables."
        self.client = OpenAI(api_key=self.api_key)
        
        self.info_path = os.path.join(BASE_DIR, f"{model.config.name}/info.json")
        self.input_path = os.path.join(BASE_DIR, f"{model.config.name}/input.jsonl")
        self.output_path = os.path.join(BASE_DIR, f"{model.config.name}/output.jsonl")
        self.results_path = os.path.join(BASE_DIR, f"{model.config.name}/results.json")
        
        os.makedirs(f"{BASE_DIR}/{model.config.name}", exist_ok=True)
    
    def generate_input(self, max_generations=200):
        
        input_items = []
        
        num_generations = 0
        num_skipped = 0
        
        for batch in self.test_loader:
            if num_generations >= max_generations:
                break
            
            sequence = batch[0].tolist()
            input_size = min(self.model.config.max_seq_len // 2, len(sequence) // 2)
            
            if input_size < 10:
                num_skipped += 1
                continue
            
            input = sequence[:input_size]
            
            if self.model.config.name == "baseline": # Baseline uses default ending
                generation = sequence[input_size:]
                if self.tokenizer.eos_token_id in generation:
                    generation = generation[:generation.index(self.tokenizer.eos_token_id)] # Remove anything after EOS token
            else:
                generation = generate_text_nucleus(self.model, self.tokenizer, input, device=self.device)
            
            assert not self.tokenizer.eos_token_id in generation, "EOS token found in generation."
            
            if len(generation) < 10: # Skip short generations (evaluation wont be meaningful)
                num_skipped += 1
                continue
            
            decoded_input = self.tokenizer.decode(input, skip_special_tokens=True)
            decoded_generation = self.tokenizer.decode(generation, skip_special_tokens=True)
        
            prompt = USER_PROMPT.replace('[STORY_BEGIN]', decoded_input).replace('[STORY_END]', decoded_generation)
            
            id = f"{self.model.config.name}_{num_generations}"
            
            input_items.append({
                    "custom_id": f"{id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000
                    }
                })
            
            num_generations += 1
        print(f"Generated {num_generations} input items (skipped {num_skipped})")

        with open(self.input_path, "w+") as f:
            for item in input_items:
                f.write(json.dumps(item) + "\n")
        print(f"Input file written to {self.input_path}")
        
    def create_batch(self):
        
        if os.path.exists(self.info_path):
            print(f"Batch info file {self.info_path} exists.")
            return
        
        if not os.path.exists(self.input_path):
            self.generate_input()
        
        input_file = self.client.files.create(file=open(self.input_path, "rb"), purpose="batch")
        
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window='24h',
            metadata={
            'description': f'{self.model.config.name} evaluation'
            }
        )
        
        with open(self.info_path, "w+") as f:
            f.write(json.dumps({"batch_id": batch.id}))
        
        print(f"Created batch with ID: {batch.id}")
    
    def save_batch_output(self):
        
        if os.path.exists(self.output_path):
            print(f"Output file {self.output_path} exists.")
            return True
        
        with open(self.info_path, "r") as f:
            info = json.load(f)
            batch_id = info["batch_id"]
            if batch_id is None:
                raise ValueError("Input batch id was not found.")
        
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(f"Batch {batch_id} completed.")
            output_id = batch.output_file_id
            output_text = self.client.files.content(output_id).text
            with open(self.output_path, "w+") as f:
                f.write(output_text)
            print(f"Output written to {self.output_path}")
            return True
        elif batch.status == "failed":
            raise ValueError(f"Batch {batch_id} failed.")
        else:
            print(f"Batch {batch_id} is still processing.")
            print(f"Status: {batch.status}")
            return False
    
    def parse_batch_output(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                return json.load(f)
        
        with open(self.output_path, "r") as f:
            lines = f.readlines()
        
        def parse_score(text, tag):
            try:
                text = text.split(f'<{tag}>')[1].split(f'</{tag}>')[0].strip()
            except:
                print(f"Error parsing {tag} from text: {text}")
                return None
            
            if '/' in text:
                text = text.split('/')[0].strip()
            
            try:
                score = int(text)
            except:
                return None
            
            return score
        
        score_map = {
            "grammar": [],
            "consistency": [],
            "plot": [],
            "creativity": []
        }
        
        num_errors = 0
        
        for line in lines:
            if not line:
                continue
            response = json.loads(line)
            custom_id = response['custom_id']
            content = response['response']['body']['choices'][0]['message']['content']
            
            grammar_score = parse_score(content, 'GRAMMAR_GRADE')
            consistency_score = parse_score(content, 'CONSISTENCY_GRADE')
            plot_score = parse_score(content, 'PLOT_GRADE')
            creativity_score = parse_score(content, 'CREATIVITY_GRADE')
            
            if None in [grammar_score, consistency_score, plot_score, creativity_score]:
                num_errors += 1
                continue
            
            score_map["grammar"].append(grammar_score)
            score_map["consistency"].append(consistency_score)
            score_map["plot"].append(plot_score)
            score_map["creativity"].append(creativity_score)
        
        print(f"Parsed {len(lines) - num_errors} scores (skipped {num_errors})")

        results = {}

        category_stats = {}
        for category in score_map:
            scores = score_map[category]
            mean = sum(scores) / len(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            category_stats[category] = {
                "mean": round(mean, 2),
                "stdev": round(std, 2)
            }

        total_scores = [
            sum(scores) / 4
            for scores in zip(
                score_map["grammar"],
                score_map["consistency"],
                score_map["plot"],
                score_map["creativity"]
            )
        ]

        total_mean = sum(total_scores) / len(total_scores)
        total_std = statistics.stdev(total_scores) if len(total_scores) > 1 else 0.0
        results["total"] = round(total_mean, 2)

        latex_string = (
            f"{round(total_mean, 2)} ({round(total_std, 2)}) & "
            f"{category_stats['grammar']['mean']} ({category_stats['grammar']['stdev']}) & "
            f"{category_stats['consistency']['mean']} ({category_stats['consistency']['stdev']}) & "
            f"{category_stats['plot']['mean']} ({category_stats['plot']['stdev']}) & "
            f"{category_stats['creativity']['mean']} ({category_stats['creativity']['stdev']})"
        )
        
        print(latex_string)

    def run_llm_eval(self):
        
        self.model.eval()
        
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        
        self.model.to(self.device)
        
        self.create_batch()
        
        if not self.save_batch_output():
            return
        
        self.parse_batch_output()
        
    def run_baseline_eval(self):
        self.create_batch()
        
        if not self.save_batch_output():
            return
        
        self.parse_batch_output()
        