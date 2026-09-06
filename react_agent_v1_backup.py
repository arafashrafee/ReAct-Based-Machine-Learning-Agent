# react_agent.py
# CSE445 Assignment #3 - Autonomous Local LLM ML Agent
# Mac-native version: Ollama served locally at 127.0.0.1:11434 (no WSL2)
#
# Task 3 additions:
#   - heal_tool_call(): controller-level self-healing for known failure modes
#   - run_agent_loop(): now catches tool exceptions, self-heals, and retries
#     once automatically, feeding the LLM an Observation that explains what
#     was corrected so its reasoning trace stays coherent.

import re
import json
import requests

from ml_tools import AVAILABLE_TOOLS

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.

You have access to the following tools:
1. load_dataset_summary(dataset_name: str) -> JSON summary of dataset (iris, wine, breast_cancer).
2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> JSON test/CV
   scores. model_type must be one of: decision_tree, logistic_regression, random_forest.
   NOTE: SVC is NOT supported here. For SVC use tune_hyperparameters instead.
3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01)
   -> JSON PyTorch training and evaluation results.
4. tune_hyperparameters(dataset_name: str, model_type: str = "svc") -> GridSearchCV results.
   model_type must be one of: svc, decision_tree.
5. reduce_features(dataset_name: str, method: str = "pca", n_components: int = 2) -> dimensionality
   reduction / feature selection results. method must be one of: pca, sfs.
6. train_advanced_pytorch_classifier(dataset_name: str, hidden_dim: int = 64, epochs: int = 50,
   lr: float = 0.01, dropout: float = 0.2) -> regularized PyTorch MLP results (BatchNorm + Dropout
   + StepLR). dropout must satisfy 0 <= dropout < 1.

To use a tool, you MUST strictly use this format:
Thought: Describe your reasoning about what to do next.
Action: <tool_name>
Action Input: {"param_name": "value"}

When you have received the observation and are ready to provide the complete answer to the user,
format your output as:
Thought: I have gathered all necessary experimental data.
Final Answer: <your complete answer>

If an Observation reports that a tool call was self-healed (corrected automatically after an
error), accept the corrected result and continue reasoning from it rather than repeating the
same mistake.

Begin!
"""


def query_local_llm(prompt: str) -> str:
    """Queries the local Ollama instance running natively on macOS."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:"],
        },
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")
    return response.json().get("response", "")


# ---------------------------------------------------------------------------
# Task 3: self-healing logic
# ---------------------------------------------------------------------------

def heal_tool_call(tool_name: str, kwargs: dict, error_or_observation):
    """
    Inspects a tool call (and, if available, its error/observation) and
    returns a corrected (new_tool_name, new_kwargs, heal_message) tuple when
    a known self-healing rule applies. Returns None if nothing needs fixing.

    error_or_observation may be:
      - None                -> pre-flight check, before the tool ever ran
      - an exception message (str) from a failed call
      - a successful JSON observation (str) that might still contain "nan"
    """
    kwargs = dict(kwargs or {})
    text = "" if error_or_observation is None else str(error_or_observation).lower()

    # Rule 1: agent tried SVC through the wrong tool -> redirect to
    # tune_hyperparameters(model_type="svc").
    if tool_name == "train_sklearn_model":
        requested_model = str(kwargs.get("model_type", "")).lower()
        if requested_model in ("svc", "svm", "support_vector_classifier", "support_vector_machine"):
            new_kwargs = {"dataset_name": kwargs.get("dataset_name"), "model_type": "svc"}
            return (
                "tune_hyperparameters",
                new_kwargs,
                "train_sklearn_model does not support SVC. Self-healed to "
                "tune_hyperparameters(model_type='svc').",
            )

    # Rule 2: reduce_features called with n_components < 1.
    if tool_name == "reduce_features":
        try:
            n_components = int(kwargs.get("n_components", 2))
        except (TypeError, ValueError):
            n_components = 0
        if n_components < 1:
            kwargs["n_components"] = 2
            return (tool_name, kwargs, "n_components < 1 is invalid. Self-healed n_components=2.")

    # Rules 3-6: train_advanced_pytorch_classifier parameter / NaN-loss errors.
    if tool_name == "train_advanced_pytorch_classifier":
        notes = []
        healed = False

        # dropout must be in [0, 1)
        try:
            dropout = float(kwargs.get("dropout", 0.2))
        except (TypeError, ValueError):
            dropout = None
        if dropout is None or not (0 <= dropout < 1):
            kwargs["dropout"] = 0.2
            notes.append("invalid dropout -> 0.2")
            healed = True

        # hidden_dim must be > 0
        try:
            hidden_dim = int(kwargs.get("hidden_dim", 64))
        except (TypeError, ValueError):
            hidden_dim = 0
        if hidden_dim <= 0:
            kwargs["hidden_dim"] = 64
            notes.append("hidden_dim <= 0 -> 64")
            healed = True

        # epochs must be > 0
        try:
            epochs = int(kwargs.get("epochs", 50))
        except (TypeError, ValueError):
            epochs = 0
        if epochs <= 0:
            kwargs["epochs"] = 80
            notes.append("epochs <= 0 -> 80")
            healed = True

        # lr must be > 0, and a NaN loss also triggers an lr/epoch/dropout reset
        try:
            lr = float(kwargs.get("lr", 0.01))
        except (TypeError, ValueError):
            lr = None
        nan_loss = "nan" in text
        if lr is None or lr <= 0 or nan_loss:
            kwargs["lr"] = 0.001
            kwargs["epochs"] = max(int(kwargs.get("epochs", 80)), 80)
            try:
                cur_dropout = float(kwargs.get("dropout", 0.2))
            except (TypeError, ValueError):
                cur_dropout = None
            if cur_dropout is None or not (0 <= cur_dropout < 1):
                kwargs["dropout"] = 0.2
            notes.append("lr <= 0 or NaN loss -> lr=0.001, dropout valid, epochs>=80")
            healed = True

        if healed:
            return (tool_name, kwargs, "Self-healed advanced PyTorch classifier params: " + "; ".join(notes))

    return None


def _execute_tool(tool_name: str, kwargs: dict) -> str:
    """
    Executes a tool with self-healing:
      1. Pre-flight heal check (catches known bad inputs before they error).
      2. Runs the tool; on exception, heals once and retries.
    Returns the text to append to the prompt as an Observation.
    """
    if tool_name not in AVAILABLE_TOOLS:
        return f"\nObservation: Tool '{tool_name}' not recognized.\n"

    # 1. Pre-flight self-heal (e.g. SVC routed to the wrong tool).
    pre = heal_tool_call(tool_name, kwargs, None)
    if pre:
        healed_tool, healed_kwargs, heal_msg = pre
        print(f"[SELF-HEAL] {heal_msg}")
        tool_name, kwargs = healed_tool, healed_kwargs

    # 2. Attempt execution, healing once on failure.
    try:
        result = AVAILABLE_TOOLS[tool_name](**kwargs)
    except Exception as exc:
        error_msg = str(exc)
        fix = heal_tool_call(tool_name, kwargs, error_msg)
        if not fix:
            return f"\nObservation: Tool execution error: {error_msg}\n"
        healed_tool, healed_kwargs, heal_msg = fix
        print(f"[SELF-HEAL] {heal_msg}")
        try:
            result = AVAILABLE_TOOLS[healed_tool](**healed_kwargs)
        except Exception as exc2:
            return f"\nObservation: Tool execution error even after self-heal: {str(exc2)}\n"
        return (
            f"\nObservation: Tool '{tool_name}' failed with error: {error_msg}. "
            f"{heal_msg} Retried successfully -> {result}\n"
        )

    # 3. Success, but still check for a NaN loss slipping through.
    fix = heal_tool_call(tool_name, kwargs, result)
    if fix:
        healed_tool, healed_kwargs, heal_msg = fix
        print(f"[SELF-HEAL] {heal_msg}")
        try:
            result = AVAILABLE_TOOLS[healed_tool](**healed_kwargs)
            return (
                f"\nObservation: Initial result contained NaN loss. {heal_msg} "
                f"Retried successfully -> {result}\n"
            )
        except Exception as exc2:
            return f"\nObservation: Tool execution error even after self-heal: {str(exc2)}\n"

    return f"\nObservation: {result}\n"


def run_agent_loop(user_query: str, max_iterations: int = 6):
    print("\n=======================================================")
    print(f"USER QUERY: {user_query}")
    print("=======================================================\n")

    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"

    for step in range(1, max_iterations + 1):
        print(f"\n--- Step {step} ---")
        llm_output = query_local_llm(prompt)
        print(llm_output)
        prompt += llm_output

        if "Final Answer:" in llm_output:
            print("\n>>> Task Completed Successfully!")
            break

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", llm_output)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_output, re.DOTALL)

        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip()
            try:
                kwargs = json.loads(raw_input)
            except json.JSONDecodeError:
                observation = "\nObservation: Error parsing Action Input as JSON.\n"
                prompt += observation
                print(observation)
                continue

            observation = _execute_tool(tool_name, kwargs)
            print(observation)
            prompt += observation
        else:
            prompt += "\nObservation: Please respond with an Action and Action Input or a Final Answer.\n"

    return prompt


if __name__ == "__main__":
    test_task = (
        "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, "
        "compare their accuracies, and recommend the best model for clinical screening."
    )
    run_agent_loop(test_task)
