from pydantic import BaseModel
import yaml
from jinja2 import Template

class PromptTemplate(BaseModel):
    name: str
    description: str
    example_output: str
    prompt: str

    def format_prompt(self, **kwargs):
        return Template(self.prompt).render(
            description=self.description,
            example_output=self.example_output,
            **kwargs
        )
    
def get_prompt(prompt_path="templates/prompt_template.yaml", example_input_path="templates/example_input.yaml"):
    with open(prompt_path) as f:
        prompt_template = yaml.safe_load(f)

    with open(example_input_path) as f:
        example_input = yaml.safe_load(f)

    prompt_template = PromptTemplate(**prompt_template)
    return prompt_template.format_prompt(**example_input)
