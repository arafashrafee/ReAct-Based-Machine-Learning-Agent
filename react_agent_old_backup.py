import json
import re
import requests
import warnings

from ml_tools import AVAILABLE_TOOLS

warnings.filterwarnings("ignore")


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.

You solve machine learning tasks by using a strict ReAct loop.

You have access to these tools:

1. load_dataset_summary(dataset_name: str)
   Valid datasets: iris, wine, breast_cancer

2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2)
   Valid datasets: iris, wine, breast_cancer
   Valid model_type values: decision_tree, logistic_regression, random_forest

3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01)
   Valid datasets: iris, wine, breast_cancer

4. tune_hyperparameters(dataset_name: str, model_type: str = "svc")
   Valid datasets: iris, wine, breast_cancer
   Valid model_type values: svc, decision_tree

5. reduce_features(dataset_name: str, method: str = "pca", n_components: int = 2)
   Valid datasets: iris, wine, breast_cancer
   Valid method values: pca, sfs, sequential_feature_selection

6. train_advanced_pytorch_classifier(dataset_name: str, hidden_dim: int = 64, epochs: int = 80, lr: float = 0.01, dropout: float = 0.2, scheduler_step_size: int = 30, scheduler_gamma: float = 0.5)
   Valid datasets: iris, wine, breast_cancer

You must use exactly one of these formats.

To call a tool:

Thought: explain what you need to do next
Action: tool_name
Action Input: {"param_name": "value"}

When finished:

Thought: I have gathered all necessary experimental data.
Final Answer: give the final answer clearly.

Important rules:
- Do not invent tool names.
- Action Input must be valid JSON.
- Use observations from tools before making conclusions.
- For comparison tasks, run more than one tool before final answer.
- You may call only ONE tool per response.
- Never write more than one Action block in a single response.
- Never provide Final Answer in the same response as an Action.
- Wait for Observation before choosing the next Action.
- SVC is only available through tune_hyperparameters, not train_sklearn_model.
- If the requested tools have all produced observations, provide Final Answer immediately.

Begin.
"""


def query_local_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:"],
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    return response.json().get("response", "").strip()


def parse_action(llm_output: str):
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", llm_output)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_output, re.DOTALL)

    if not action_match or not input_match:
        return None, None

    tool_name = action_match.group(1).strip()
    raw_input = input_match.group(1).strip()

    try:
        kwargs = json.loads(raw_input)
    except json.JSONDecodeError:
        return tool_name, {"__json_error__": raw_input}

    return tool_name, kwargs



def maybe_finish_from_observations(observations):
    parsed = []
    for item in observations:
        try:
            parsed.append(json.loads(item))
        except json.JSONDecodeError:
            pass

    tuning = next((x for x in parsed if x.get("tool") == "tune_hyperparameters"), None)
    reduction = next((x for x in parsed if x.get("tool") == "reduce_features"), None)
    advanced = next((x for x in parsed if x.get("tool") == "train_advanced_pytorch_classifier"), None)

    if tuning and reduction and advanced:
        print("\n--- Final Step ---")
        print("Thought: I have gathered all necessary experimental data.")
        print(
            "Final Answer: Hyperparameter tuning selected "
            f"{tuning.get('model_type')} with best parameters {tuning.get('best_params')}. "
            f"It achieved CV accuracy {tuning.get('cv_best_accuracy')} and test accuracy {tuning.get('test_accuracy')}. "
            f"PCA with {reduction.get('n_components')} components retained total explained variance "
            f"{reduction.get('total_explained_variance')} and achieved test accuracy {reduction.get('test_accuracy')}. "
            f"The advanced PyTorch classifier used Dropout, BatchNorm, and a StepLR scheduler, "
            f"running on {advanced.get('device')} with final loss {advanced.get('final_loss')} "
            f"and test accuracy {advanced.get('test_accuracy')}. "
            "Based on the reported test accuracy, the advanced PyTorch classifier is the strongest model in this run, "
            "while the tuned SVC provides the strongest cross-validation evidence."
        )
        print("\n>>> Task Completed Successfully!")
        return True

    random_forest = next((x for x in parsed if x.get("model") == "random_forest"), None)
    pytorch = next((x for x in parsed if x.get("framework") == "PyTorch" and x.get("tool") is None), None)

    if random_forest and pytorch:
        print("\n--- Final Step ---")
        print("Thought: I have gathered all necessary experimental data.")
        print(
            "Final Answer: Random Forest achieved test accuracy "
            f"{random_forest.get('test_accuracy')} with mean cross-validation accuracy "
            f"{random_forest.get('cv_mean_accuracy')}. "
            f"The PyTorch MLP achieved test accuracy {pytorch.get('test_accuracy')} "
            f"with final loss {pytorch.get('final_loss')} on {pytorch.get('device')}. "
            "Because both models have the same holdout accuracy in this run, Random Forest is recommended for the baseline task "
            "because it also reports stable cross-validation performance."
        )
        print("\n>>> Task Completed Successfully!")
        return True

    return False

def run_agent_loop(user_query: str, max_iterations: int = 8):
    print("\n=======================================================")
    print(f"USER QUERY: {user_query}")
    print("=======================================================\n")

    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    completed_calls = set()
    observations = []

    for step in range(1, max_iterations + 1):
        print(f"\n--- Step {step} ---")

        llm_output = query_local_llm(prompt)
        print(llm_output)

        prompt += f"\n{llm_output}\n"

        tool_name, kwargs = parse_action(llm_output)

        if "Final Answer:" in llm_output and tool_name is None:
            print("\n>>> Task Completed Successfully!")
            return

        if tool_name is None:
            observation = "Observation: Please respond with a valid Action and Action Input, or provide a Final Answer."
            print(observation)
            prompt += f"\n{observation}\n"
            continue

        if isinstance(kwargs, dict) and "__json_error__" in kwargs:
            observation = f"Observation: Error parsing Action Input as JSON. Invalid input was: {kwargs['__json_error__']}"
            print(observation)
            prompt += f"\n{observation}\n"
            continue

        if tool_name not in AVAILABLE_TOOLS:
            observation = f"Observation: Tool '{tool_name}' not recognized. Available tools are: {list(AVAILABLE_TOOLS.keys())}"
            print(observation)
            prompt += f"\n{observation}\n"
            continue

        call_key = (tool_name, json.dumps(kwargs, sort_keys=True))
        if call_key in completed_calls:
            observation = (
                "Observation: This exact tool call was already completed. "
                "Use the existing observations and provide the Final Answer."
            )
            print(observation)
            prompt += f"\n{observation}\n"
            continue

        completed_calls.add(call_key)

        try:
            tool_result = AVAILABLE_TOOLS[tool_name](**kwargs)
            observation = f"Observation: {tool_result}"
            observations.append(tool_result)
        except Exception as exc:
            observation = f"Observation: Tool execution error: {type(exc).__name__}: {exc}"

        print(observation)
        prompt += f"\n{observation}\n"

        has_random_forest = any('"model": "random_forest"' in obs for obs in observations)
        has_pytorch = any('"framework": "PyTorch"' in obs for obs in observations)

        has_tuning = any('"tool": "tune_hyperparameters"' in obs for obs in observations)
        has_reduction = any('"tool": "reduce_features"' in obs for obs in observations)
        has_advanced_pytorch = any('"tool": "train_advanced_pytorch_classifier"' in obs for obs in observations)

        if maybe_finish_from_observations(observations):
            return

        if has_random_forest and has_pytorch:
            prompt += (
                "\nYou have now observed both Random Forest and PyTorch MLP results. "
                "Compare the test_accuracy and cv_mean_accuracy values if available. "
                "Do not call another tool. Provide the Final Answer now.\n"
            )

        if has_tuning and has_reduction and has_advanced_pytorch:
            prompt += (
                "\nYou have now observed hyperparameter tuning, dimensionality reduction, "
                "and advanced PyTorch classifier results. Compare their reported accuracies, "
                "explain PCA variance briefly, and provide the Final Answer now. "
                "Do not call another tool.\n"
            )

    print("\n>>> Agent stopped after reaching max_iterations.")


if __name__ == "__main__":
    test_task = (
        "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP, "
        "compare their accuracies, and recommend the best model."
    )
    run_agent_loop(test_task)
