from threading import Lock

class LLM:
    def __init__(self, model, default_seed=42):
        self.model = model
        self.lock = Lock()
        self.default_seed = default_seed
        
    def create_evaluation_prompt(self, question, system_msg=None):
        if system_msg is None:
            system_msg = "You are a general assistant"
        
        prompt = f"""
<|start_header_id|>system<|end_header_id|>
{system_msg}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

    def generate(self, prompt, max_tokens=512, temperature=0.1, seed=None):
        with self.lock:
            self.model.reset()
            
            # Use provided seed or default
            current_seed = seed if seed is not None else self.default_seed
            
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,  # from 0.0 to 0.9 (1.0 = off)
                top_k=40,   
                repeat_penalty=1.1,  
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["<|eot_id|>"],
                echo=False,
                seed=current_seed  # Seed ensures reproducibility
            )
            
            text = result["choices"][0]["text"].strip()
            
            # Remove any trailing incomplete sentences
            if text and text.strip() and text[-1] not in '.!?"\'\)]}':
                # Find last complete sentence
                last_period = text.rfind('.')
                last_exclamation = text.rfind('!')
                last_question = text.rfind('?')
                
                last_complete = max(last_period, last_exclamation, last_question)
                if last_complete != -1:
                    text = text[:last_complete + 1]
            
            return text