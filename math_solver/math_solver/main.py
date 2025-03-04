#!/usr/bin/env python3
"""
Math Solver AI using Gradio interface.

Before running, install the required packages:
    pip install transformers gradio torch langchain-ollama

Additionally, ensure you have set up ollama (see https://ollama.ai/install.sh for instructions).
"""

import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM

# Initialize the model with the recommended temperature
model = OllamaLLM(model="erwan2/DeepSeek-R1-Distill-Qwen-7B:latest", base_url="http://ollama:11434", temperature=0.6)


def format_response(response):
    """Formats the response to properly use LaTeX math notation."""
    response = response.replace("$", "\\(").replace("$$", "\\[\n")  # Inline and block math
    response = response.replace("\\(", "\\(").replace("\\[", "\\[")  # Ensure proper format
    response = response.replace("\\]", "\\]\n")  # Close display math properly
    return response


# Few-shot examples for better structured responses
FEW_SHOT_EXAMPLES = """
### Example 1: Solving a Linear Equation
**User:** Solve for x: 2x + 5 = 15
**AI:**
<think>
1. Subtract 5 from both sides: 2x = 10
2. Divide by 2: x = 5
</think>
**Final Answer:** \\boxed{5}

### Example 2: Solving a Quadratic Equation
**User:** Solve for x: x² - 5x + 6 = 0
**AI:**
<think>
1. Factor the quadratic equation: (x - 3)(x - 2) = 0
2. Solve for x: x = 3 or x = 2
</think>
**Final Answer:** \\boxed{3, 2}

### Example 3: Finding a Derivative
**User:** Differentiate f(x) = 3x² + 4x - 7
**AI:**
<think>
1. Use the power rule: d/dx [ax^n] = n * ax^(n-1)
2. Differentiate each term:
   - d/dx [3x²] = 6x
   - d/dx [4x] = 4
   - d/dx [-7] = 0
</think>
**Final Answer:** \\boxed{6x + 4}
"""


def chatbot(question):
    """
    Process the user's question and return a step-by-step explanation and final answer.
    """
    # Structure the prompt with step-by-step reasoning enforcement
    formatted_question = (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "<think>\n"
        f"{question}\n"
        "</think>\n"
        "Please reason step by step and provide the final answer inside \\boxed{}."
    )

    messages = [HumanMessage(content=formatted_question)]
    response = model.invoke(messages).strip()
    return format_response(response)


# Gradio interface definition
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="📚 Math Solver AI",
    description="Solve algebra, calculus, and probability problems with step-by-step explanations.",
)

# if __name__ == "__main__":
#     iface.launch(server_name="0.0.0.0", server_port=7861, debug=True, share=True)
iface.launch(server_name="0.0.0.0", server_port=7861, debug=True, share=True)
