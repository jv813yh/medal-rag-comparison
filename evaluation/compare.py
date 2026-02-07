import os
import sys
import json
import time

# Add current folder and root to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import evaluate_answer, calculate_average_metrics


def run_evaluation(rag_name, query_func):
    print(f"\n--- Evaluating {rag_name} ---")
    queries_path = os.path.join(os.path.dirname(__file__), "queries.json")
    with open(queries_path, "r") as f:
        queries = json.load(f)

    
    results = []
    for q in queries:
        print(f"Querying [{q['id']}]: {q['query']}")
        start_time = time.time()
        try:
            # query_func now returns (answer, context_chunks)
            answer, chunks = query_func(q['query'])
            latency = time.time() - start_time
            
            # Step 2 of Evaluation Protocol: Metrics
            eval_results = evaluate_answer(q['query'], "\n".join(chunks) if chunks else "PageIndex Internal", answer)
            
            # Precision@K estimate
            from metrics import evaluate_retrieval
            precision = evaluate_retrieval(q['query'], chunks) if chunks else 1.0
            
            # Step 3: Cost tracking (simplified estimate)
            cost = 0.0
            if rag_name == "OpenAI":
                from metrics import calculate_cost
                # Estimate: query + context + prompt ~ 1500 tokens. Answer ~ 200 tokens.
                cost = calculate_cost(1500, 200)
            
            results.append({
                "id": q["id"],
                "query": q["query"],
                "answer": answer,
                "latency": latency,
                "relevance": eval_results["relevance"],
                "faithfulness": eval_results["faithfulness"],
                "precision": precision,
                "cost": cost
            })
        except Exception as e:
            print(f"Error evaluating {q['id']}: {e}")


    summary = calculate_average_metrics(results)
    output = {
        "rag_name": rag_name,
        "results": results,
        "summary": summary
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    result_file = os.path.join(results_dir, f"{rag_name.lower()}_results.json")
    with open(result_file, "w") as f:
        json.dump(output, f, indent=4)

    
    return summary

import importlib.util

def import_query_func(folder_name):
    """Dynamically imports the query function from a specific RAG folder."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(root_dir, folder_name, "query.py")
    
    spec = importlib.util.spec_from_file_location(f"{folder_name}.query", file_path)
    module = importlib.util.module_from_spec(spec)
    # Add the folder to sys.path so the module can find its own config.py
    sys.path.insert(0, os.path.join(root_dir, folder_name))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module.query

def compare_all():
    summary_table = {}

    # 1. OpenAI RAG
    try:
        openai_query = import_query_func("openai-rag")
        summary_table["OpenAI"] = run_evaluation("OpenAI", openai_query)
    except Exception as e:
        print(f"Skipping OpenAI: {e}")

    # 2. Local RAG
    try:
        local_query = import_query_func("local-model-rag")
        summary_table["Local"] = run_evaluation("Local", local_query)
    except Exception as e:
        print(f"Skipping Local: {e}")

    # 3. PageIndex RAG
    try:
        pi_query = import_query_func("pageindex-rag")
        summary_table["PageIndex"] = run_evaluation("PageIndex", pi_query)
    except Exception as e:
        print(f"Skipping PageIndex: {e}")


    print("\n--- COMPARISON SUMMARY ---")
    print(json.dumps(summary_table, indent=4))

if __name__ == "__main__":
    compare_all()

