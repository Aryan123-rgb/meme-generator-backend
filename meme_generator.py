from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import os
from pydantic import BaseModel, Field
import aiohttp
import json
import random
from dotenv import load_dotenv

load_dotenv()

# Advanced graph states
class MemeConcept(BaseModel):
    message: str = Field(description="The core message of the meme")
    emotion: str = Field(description="The emotion conveyed by the meme")

class MemeConcepts(BaseModel):
    concepts: List[MemeConcept] = Field(description="List of meme concepts")

class TemplateInfo(BaseModel):
    template_id: str = Field(..., description="Unique identifier for the template")
    name: str = Field(..., description="Name of the meme template")
    blank_template_api_link: str = Field(..., description="API link to the blank template")
    description: str = Field(..., description="Description of the meme template")
    example_text_1: Optional[str] = Field("", description="Example text for the first line")
    example_text_2: Optional[str] = Field("", description="Example text for the second line")
    lines: int = Field(..., description="Number of text lines in the meme")
    keywords: List[str] = Field(..., description="Keywords associated with the template")

class SelectedMeme(BaseModel):
    meme_id: str = Field(..., description="Unique identifier for the selected meme")
    template_id: str = Field(..., description="ID of the selected template")
    concept: MemeConcept = Field(..., description="The concept associated with the meme")
    example_text_1: Optional[str] = Field("", description="Example text for the first line")
    example_text_2: Optional[str] = Field("", description="Example text for the second line")
    template_info: TemplateInfo = Field(..., description="Information about the selected template")
    blank_template_api_link: str = Field(..., description="API link to the blank template without extension")

class PreGeneratedMeme(BaseModel):
    meme_id: str = Field(..., description="Unique identifier for the selected meme")
    blank_template_api_link: str = Field(..., description="API link to the blank template without extension")
    generated_text_element1: str = Field(..., description="Generated text element 1")
    generated_text_element2: str = Field(..., description="Generated text element 2")

# Graph State
class GraphState(TypedDict):
    """Enhanced state object for the meme generation workflow."""
    user_message: str
    meme_concepts: Annotated[List[Dict], "Generated meme concepts"]
    selected_memes: Annotated[Dict[str, SelectedMeme], "Selected memes with their info"]
    available_templates: Annotated[Dict[str, TemplateInfo], "Available meme templates"]
    pre_generated_memes: Annotated[Dict[str, PreGeneratedMeme], "Pre-Generated memes with their info"]
    generated_memes: Annotated[List[str], "Generated memes with their info"]

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def generate_meme_concepts(state: GraphState) -> GraphState:
    """Generate meme concepts using structured output."""
    structured_llm = llm.with_structured_output(MemeConcepts)
    prompt = f"""Based on this user request: "{state['user_message']}"
    
Create 3 creative meme concepts with different messages and emotions."""
    
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        memes = result.model_dump()
        state["meme_concepts"] = memes.get('concepts', [])
    except Exception as e:
        print(f"Error generating concepts: {e}")
        state["meme_concepts"] = []
    
    return state


async def get_meme_templates() -> Dict[str, TemplateInfo]:
    """
    Fetches available meme templates from Memegen.link API.

    Returns:
        Dict[str, TemplateInfo]: Dictionary of template information keyed by template ID

    Notes:
        - Fetches templates from Memegen.link API
        - Filters templates to those with 2 or fewer lines
        - Randomly selects 20 templates
        - Converts API response to TemplateInfo objects
        - Includes template metadata like name, description, and example text
    """

    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.memegen.link/templates/") as response:
            all_templates = await response.json()

            # Filter templates with 2 or fewer lines
            filtered_templates = [
                template for template in all_templates
                if template.get("lines", 0) <= 2
            ]

            # Select 20 random templates
            selected_templates = random.sample(filtered_templates, min(30, len(filtered_templates)))

            # Convert to dictionary with template ID as key, mapping fields to TemplateInfo
            template_dict = {
                template["id"]: TemplateInfo(
                    template_id=template["id"],
                    name=template["name"],
                    blank_template_api_link=template["blank"],
                    description=f"{template['name']} meme with {template['lines']} text lines.",
                    example_text_1=template.get('example', {}).get('text', [''])[0] or '',
                    example_text_2=template.get('example', {}).get('text', ['', ''])[1] if len(template.get('example', {}).get('text', [])) > 1 else '',
                    lines=template["lines"],
                    keywords=template.get("keywords", [])
                )
                for template in selected_templates
            }
            return template_dict


def select_meme_templates(state: GraphState) -> GraphState:
    """
    Selects appropriate meme templates for each concept.

    Args:
        state (GraphState): Current state containing selected_concepts and available_templates

    Returns:
        GraphState: Updated state with selected_memes added

    Notes:
        - Creates simplified template descriptions for LLM
        - Matches concepts with appropriate templates
        - Falls back to random selection if no match found
        - Handles template selection for each concept
        - Creates structured meme objects with template info
    """

    concepts = state['meme_concepts']
    templates = state["available_templates"]
    selected_memes = {}

    # Create simplified template descriptions for the LLM
    template_descriptions = [
        {
            'template_id': template_id,
            'name': template_data.name,
            'description': template_data.description,
            'lines': template_data.lines,
            "example_text_1": template_data.example_text_1,
            "example_text_2": template_data.example_text_2
        }
        for template_id, template_data in templates.items()
    ]

    for idx, concept in enumerate(concepts):
        prompt = f"""Select a meme template that best fits this concept:

        Concept:
        - Message: {concept['message']}
        - Emotion: {concept['emotion']}

        Available Templates:
        {json.dumps(template_descriptions, indent=2)}

        Return only the template ID that best matches the concept's message and emotion."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            # Extract template ID from response, removing quotes and whitespace
            template_id = response.content.strip().strip('"').strip("'").lower()

            # Fallback to random template if not found
            if template_id not in templates:
                template_id = random.choice(list(templates.keys()))

            # Create meme object
            selected_memes[f"meme_{idx+1}"] = {
                "meme_id": f"meme_{idx+1}",
                "template_id": template_id,
                "concept": concept,
                "template_info": templates[template_id],
                "blank_template_api_link": templates[template_id].blank_template_api_link,
                "example_text_1": templates[template_id].example_text_1,
                "example_text_2": templates[template_id].example_text_2,
            }

        except Exception as e:
            print(f"Error selecting template: {str(e)}")
            continue

    state["selected_memes"] = selected_memes
    return state


def generate_text_elements(state: GraphState) -> GraphState:
    """
    Generates meme text based on selected concepts and templates.

    Args:
        state (GraphState): Current state containing selected_memes and company_context

    Returns:
        GraphState: Updated state with pre_generated_memes added

    Notes:
        - Generates appropriate text for each template
        - Considers template format and number of lines
        - Maintains brand tone and target audience
        - Creates concise, punchy text elements
        - Handles errors gracefully for each meme
    """

    selected_memes = state["selected_memes"]
    pre_generated_memes = {}

    for meme_id, meme in selected_memes.items():
        concept = meme["concept"]
        template_info = meme["template_info"]

        prompt = f"""Create text for a meme based on this template and concept:

        Template: {template_info.name}
        Number of lines: {template_info.lines}
        Description: {template_info.description}
        Example Text 1: {template_info.example_text_1}
        Example Text 2: {template_info.example_text_2}
        Concept Message: {concept['message']}
        Emotion: {concept['emotion']}
        
        Return ONLY the text lines, one per line. Keep each line concise and punchy.
        """

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text_elements = response.content.strip().split('\n')

            generated_text1 = text_elements[0] if len(text_elements) > 0 else ""
            generated_text2 = text_elements[1] if len(text_elements) > 1 else ""

            pre_generated_memes[meme_id] = {
                "meme_id": meme_id,
                "blank_template_api_link": meme['blank_template_api_link'],
                "generated_text_element1": generated_text1,
                "generated_text_element2": generated_text2
            }

        except Exception as e:
            print(f"Error generating text: {str(e)}")
            continue

    state["pre_generated_memes"] = pre_generated_memes
    return state


import os
import re

def create_meme_url(state: GraphState) -> GraphState:
    """
    Creates final meme URLs using the Memegen.link API format.

    Args:
        state (GraphState): Current state containing pre_generated_memes

    Returns:
        GraphState: Updated state with generated_memes added

    Notes:
        - Processes text elements for URL compatibility
        - Handles URL encoding of text
        - Extracts and manages file extensions
        - Constructs final meme URLs
        - Maintains all meme metadata in state
    """

    pre_generated_memes = state["pre_generated_memes"]
    generated_memes = []

    for meme_id, meme in pre_generated_memes.items():
        # Replace spaces with underscores
        text1 = meme["generated_text_element1"].replace(' ', '_')
        text2 = meme["generated_text_element2"].replace(' ', '_')

        # Remove trailing ? or . and keep only letters and underscores
        text1 = re.sub(r'[^A-Za-z_]', '', text1.rstrip('?.'))
        text2 = re.sub(r'[^A-Za-z_]', '', text2.rstrip('?.'))

        # Get template info
        base_url = meme['blank_template_api_link']

        # Extract extension
        extension = os.path.splitext(base_url)[1]
        base_url = base_url.rsplit('.', 1)[0]

        # Construct final URL
        final_url = f"{base_url}/{text1}/{text2}{extension}"
        generated_memes.append(final_url)

    state['generated_memes'] = generated_memes
    return state



graph = StateGraph(GraphState)

graph.add_node("generate_meme_concepts", generate_meme_concepts)
graph.add_node("select_meme_templates", select_meme_templates)
graph.add_node("generate_text_elements", generate_text_elements)
graph.add_node("create_meme_url", create_meme_url)

graph.add_edge(START, "generate_meme_concepts")
graph.add_edge("generate_meme_concepts", "select_meme_templates")
graph.add_edge("select_meme_templates", "generate_text_elements")
graph.add_edge("generate_text_elements", "create_meme_url") 
graph.add_edge("create_meme_url", END)

workflow = graph.compile()

async def run_workflow(user_query: str) -> list[str]:
    available_templates = await get_meme_templates()
    initial_state = {
        "user_message": "AI is going to take over the world",
        "available_templates": available_templates
    }
    output_state = await workflow.ainvoke(initial_state)
    print(output_state['generated_memes'])
    return output_state['generated_memes']