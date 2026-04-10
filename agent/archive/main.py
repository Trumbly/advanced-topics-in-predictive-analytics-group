import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLMClient
from code_runner import CodeRunner
from experiment_log import ExperimentLog
from prompt_builder import PromptBuilder

def run_agent(num_iterations=10, model="gemma4:e4b"):
    llm = LLMClient(model_name=model)
    runner = CodeRunner(timeout_seconds=300)
    log = ExperimentLog(log_dir="experiments")
    builder = PromptBuilder()

    print("=" * 60)
    print("BIRDCLEF AI AGENT STARTING")
    print(f"Model: {model} | Iterations: {num_iterations}")
    print("=" * 60)

    with open("data/data_summary.json") as f:
        data_info = json.load(f)

    best_score = 0.0

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- EXPERIMENT {iteration}/{num_iterations} ---")

        # Step 1: Ask LLM to propose experiment
        prompt = builder.build_proposal_prompt(
            data_info=data_info,
            past_experiments=log.get_summary(),
            iteration=iteration
        )
        print("Asking Gemma 4 to propose an experiment...")
        proposal = llm.ask(prompt)
        print(f"Proposal: {proposal[:200]}...")

        # Step 2: Generate code
        print("Generating Python training code...")
        code_prompt = builder.build_code_prompt(proposal, data_info)
        raw_response = llm.ask(code_prompt)
        generated_code = llm.extract_code(raw_response)

        # Step 3: Save to file
        exp_dir = log.create_experiment_dir(iteration)
        script_path = f"{exp_dir}/train.py"
        with open(script_path, "w") as f:
            f.write(generated_code)
        print(f"Code saved to {script_path}")

        # Step 4: Run the code
        print("Running generated code...")
        result = runner.run(script_path)

        # Step 5: Show results
        if result["success"]:
            score = result.get("roc_auc", 0.0)
            print(f"SUCCESS — ROC-AUC: {score:.4f}")
            if score > best_score:
                best_score = score
                print(f"NEW BEST SCORE!")
        else:
            score = 0.0
            print(f"FAILED — {result['error'][:200]}")

        # Step 6: Ask LLM to analyze
        analysis = llm.ask(builder.build_analysis_prompt(proposal, generated_code, result))

        # Step 7: Log everything
        log.save_experiment({
            "iteration": iteration,
            "proposal": proposal,
            "code": generated_code,
            "result": result,
            "analysis": analysis,
            "score": score,
            "best_score": best_score
        }, exp_dir)

        print(f"Best overall: {best_score:.4f}")
        time.sleep(1)

    print(f"\nAGENT DONE. Best ROC-AUC: {best_score:.4f}")

if __name__ == "__main__":
    run_agent(num_iterations=10, model="gemma4:e4b")
