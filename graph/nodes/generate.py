from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    generation = generation_chain.invoke(
        {"context": state["documents"], "question": state["question"]}
    )
    return {"generation": generation}
