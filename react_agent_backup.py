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

CRITICAL RULE: Produce exactly ONE Thought, ONE Action, and ONE Action Input, then STOP
immediately. Do NOT write "Observation:" yourself, and do NOT write a second "Thought:" in the
same turn. The real Observation will be given to you by the system after the tool actually runs.

Example of a single correct turn (stop right after the closing brace):
Thought: I need to check what the iris dataset looks like before training anything.
Action: load_dataset_summary
Action Input: {"dataset_name": "iris"}

IMPORTANT - do not pre-correct parameters yourself: If the user's instructions ask you to try a
specific tool or a specific parameter value (even one you suspect is unsupported or invalid, such
as model_type="svc" for train_sklearn_model, or dropout=1.5), call the tool exactly as instructed
FIRST. Do not silently substitute a "better" tool or a "corrected" value in your own Thought and
skip the call. The system has automatic self-healing: it will catch the real error from the tool
and either redirect you to the correct tool or correct the invalid parameter for you, then give you
a real Observation showing what was healed. Only adjust your own approach after you have seen that
real Observation.

Do not repeat an Action with the exact same Action Input you already ran and already received an
Observation for — move on to the next step of the task instead.

If an Observation reports that a tool call was self-healed (corrected automatically after an
error), accept the corrected result and continue reasoning from it rather than repeating the
same mistake.

Begin!
"""


def _truncate_to_first_turn(text: str) -> str:
    """
    Keep only the FIRST Thought/Action/Action-Input block (or the first
    Final Answer, if the model decides no tool call is needed).

    Small local models frequently ignore the "stop" sequence sent to Ollama
    and keep hallucinating several fake steps (including fake Observations)
    in a single generation. Relying on the API's stop sequence alone is not
    reliable enough, so the controller enforces single-step behaviour itself
    by truncating the raw text before it is ever appended to the prompt or
    parsed for a tool call. This guarantees the agent only ever acts on real,
    freshly-executed Observations.
    """
    action_idx = text.find("Action:")
    final_idx = text.find("Final Answer:")

    # If the model answered directly (no tool call), keep just that answer —
    # but still cut off anything that follows (a stray Action/Thought the
    # model tacked on after its own Final Answer).
    if final_idx != -1 and (action_idx == -1 or final_idx < action_idx):
        next_thought = text.find("Thought:", final_idx + len("Final Answer:"))
        next_action = text.find("Action:", final_idx + len("Final Answer:"))
        candidates = [i for i in (next_thought, next_action) if i != -1]
        cut = min(candidates) if candidates else len(text)
        return text[:cut]

    if action_idx == -1:
        return text

    ai_idx = text.find("Action Input:", action_idx)
    if ai_idx == -1:
        return text

    brace_start = text.find("{", ai_idx)
    if brace_start == -1:
        return text

    # Brace-match to find the end of the first complete JSON object, so
    # nested braces inside Action Input are handled correctly.
    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]

    return text


def query_local_llm(prompt: str) -> str:
    """Queries the local Ollama instance running natively on macOS."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            # Stop as soon as the model finishes ONE Action Input, before it can
            # hallucinate a fake Observation or jump straight into a second
            # Thought/Action pair. This forces genuine turn-by-turn tool calls
            # instead of the model faking the whole trace in one generation.
            "stop": ["Observation:", "\nThought:", "\n\nThought:"],
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


def _parse_inline_call_args(llm_output: str, tool_name: str) -> dict:
    """
    Fallback parser: some local models write arguments inline in the Action
    line itself, e.g. `Action: load_dataset_summary(dataset_name="iris")`,
    instead of putting them in Action Input as JSON. If Action Input came
    back empty, recover kwargs from this inline call syntax so the tool
    still gets real arguments instead of failing on missing required params.
    """
    pattern = rf"Action:\s*{re.escape(tool_name)}\s*\((.*?)\)"
    match = re.search(pattern, llm_output)
    if not match:
        return {}

    args_str = match.group(1).strip()
    if not args_str:
        return {}

    kwargs = {}
    for part in args_str.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val.lower() in ("true", "false"):
            kwargs[key] = val.lower() == "true"
            continue
        try:
            kwargs[key] = float(val) if "." in val else int(val)
        except ValueError:
            kwargs[key] = val
    return kwargs


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


def _extract_explicit_calls(user_query: str):
    """
    Parses literal `tool_name(key='value', key2=value2, ...)` calls mentioned
    in the task text, in the order they appear.

    Used as a stuck-loop autopilot: if the local LLM keeps repeating the same
    action instead of progressing (a real limitation of small local models
    over long contexts), the controller can execute the NEXT explicitly
    requested step itself, so the self-healing mechanism still gets
    exercised end-to-end instead of the run silently looping forever.
    """
    calls = []
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)", user_query):
        tool_name = m.group(1)
        if tool_name not in AVAILABLE_TOOLS:
            continue
        kwargs = {}
        for part in m.group(2).split(","):
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() in ("true", "false"):
                kwargs[key] = val.lower() == "true"
                continue
            try:
                kwargs[key] = float(val) if "." in val else int(val)
            except ValueError:
                kwargs[key] = val
        calls.append((tool_name, kwargs))
    return calls


def run_agent_loop(user_query: str, max_iterations: int = 6):
    print("\n=======================================================")
    print(f"USER QUERY: {user_query}")
    print("=======================================================\n")

    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    seen_calls = {}  # (tool_name, frozenset(kwargs.items())) -> observation text
    explicit_calls = _extract_explicit_calls(user_query)
    explicit_call_idx = 0

    for step in range(1, max_iterations + 1):
        print(f"\n--- Step {step} ---")
        raw_output = query_local_llm(prompt)
        llm_output = _truncate_to_first_turn(raw_output)
        if llm_output != raw_output:
            print("[CONTROLLER] Model kept generating past one step; truncated to the first "
                  "Thought/Action/Action Input (or first Final Answer).")
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

            if not kwargs:
                fallback_kwargs = _parse_inline_call_args(llm_output, tool_name)
                if fallback_kwargs:
                    print(f"[CONTROLLER] Action Input was empty; recovered arguments from "
                          f"inline call syntax: {fallback_kwargs}")
                    kwargs = fallback_kwargs

            call_key = (tool_name, frozenset(kwargs.items()))
            if call_key in seen_calls:
                # The model is stuck repeating a call it already made. Instead of
                # burning another iteration hoping it self-corrects, the controller
                # auto-advances to the next explicitly requested step (if any),
                # running it through the same self-healing path.
                while explicit_call_idx < len(explicit_calls):
                    next_tool, next_kwargs = explicit_calls[explicit_call_idx]
                    explicit_call_idx += 1
                    next_key = (next_tool, frozenset(next_kwargs.items()))
                    if next_key not in seen_calls:
                        break
                else:
                    next_tool, next_kwargs, next_key = None, None, None

                if next_tool is not None and next_key not in seen_calls:
                    print(f"[CONTROLLER] Stuck-loop detected repeating {tool_name}{dict(kwargs)}. "
                          f"Auto-advancing to the next explicitly requested step: "
                          f"{next_tool}({next_kwargs})")
                    auto_observation = _execute_tool(next_tool, next_kwargs)
                    seen_calls[next_key] = auto_observation.replace("\nObservation: ", "").strip()
                    observation = (
                        f"\nObservation: You already ran {tool_name} with these exact arguments; "
                        f"no need to repeat it. Since you were not progressing, the controller "
                        f"executed the next required step on your behalf:\n"
                        f"{next_tool}({next_kwargs}) -> {seen_calls[next_key]}\n"
                        f"Continue reasoning from this new result. If you now have everything "
                        f"you need, give your Final Answer.\n"
                    )
                else:
                    observation = (
                        f"\nObservation: You already ran {tool_name} with these exact arguments. "
                        f"Reusing the previous result instead of re-running it: "
                        f"{seen_calls[call_key]}\n"
                        f"Move on to the next step, or give your Final Answer if you have "
                        f"everything you need.\n"
                    )
                    print("[CONTROLLER] Duplicate tool call detected; reusing cached result.")
            else:
                observation = _execute_tool(tool_name, kwargs)
                seen_calls[call_key] = observation.replace("\nObservation: ", "").strip()

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
