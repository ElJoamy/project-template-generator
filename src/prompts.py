from enum import Enum
from pydantic import BaseModel, Field

PROJECT_TEMPLATE = """You are an {major} senior student looking for a final project.
    In order to classify for your dissertation you need to present a project proposal with the following
    parts: 
    - Title
    - Hypothesis
    - Problem to solve
    - Justification
    - Main objective
    Generate {n_examples} examples for final projects with the given format in {language} language
    using the {format_instructions}.
    """


class Major(str, Enum):
    civil_engineering = "civil engineering"
    electrical_computer_engineering = "electrical and computer engineering"
    computer_science = "computer science"
    industrial_engineering = "industrial engineering"
    chemical_engineering = "chemical engineering"


class Language(str, Enum):
    spanish = "spanish"
    english = "english"


class ProjectParams(BaseModel):
    major: Major
    language: Language
    n_examples: int
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # donde default es el valor por defecto, ge es mayor o igual y le es menor o igual