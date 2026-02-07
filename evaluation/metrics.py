import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_answer(query, context, answer):
    """
    Uses LLM-as-a-judge to score the answer's relevance and faithfulness.
    Returns a dict with scores (0-10).
    """
    prompt = f"""
    Evaluate the following RAG system response based on the provided context and query.
    
    Query: {query}
    Context: {context}
    Answer: {answer}
    
    Provide a score from 0 to 10 for:
    1. Relevance: How well the answer addresses the query.
    2. Faithfulness: How accurately the answer reflects the provided context (no hallucinations).
    
    Format:
    Relevance: [score]
    Faithfulness: [score]
    Reasoning: [short explanation]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a rigorous evaluation judge."},
                      {"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        
        # Simple parsing logic
        relevance = 0
        faithfulness = 0
        for line in content.split('\n'):
            if "Relevance:" in line:
                relevance = float(line.split(':')[1].strip().split(' ')[0])
            if "Faithfulness:" in line:
                faithfulness = float(line.split(':')[1].strip().split(' ')[0])
                
        return {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "raw_eval": content
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {"relevance": 0, "faithfulness": 0, "error": str(e)}

def calculate_cost(tokens_in, tokens_out, model="gpt-4o-mini"):
    """
    Step 3 of Evaluation Protocol: Track cost ($).
    """
    if model == "gpt-4o-mini":
        return (tokens_in * 0.15 / 1000000) + (tokens_out * 0.60 / 1000000)
    elif model == "text-embedding-3-small":
        return tokens_in * 0.02 / 1000000
    return 0.0

def evaluate_retrieval(query, chunks):
    """
    Step 2 of Evaluation Protocol: Retrieval Precision@K.
    Uses LLM to check if the TOP-K chunks are actually relevant.
    """
    chunks_text = "\n---\n".join([f"Chunk {i}: {c}" for i, c in enumerate(chunks)])
    prompt = f"""
    Query: {query}
    Retrieved Chunks:
    {chunks_text}
    
    For each chunk, determine if it contains information relevant to the query.
    Return a list of boolean values (True/False) only.
    Format: [True, False, True...]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        # Simple extraction of bools
        import ast
        try:
            results = ast.literal_eval(content.strip())
            precision = sum(results) / len(results) if results else 0
            return precision
        except:
            return 0.5 # Default if parsing fails
    except Exception as e:
        return 0.0

def calculate_average_metrics(results_list):

    if not results_list:
        return {}
    
    avg_relevance = sum(r['relevance'] for r in results_list) / len(results_list)
    avg_faithfulness = sum(r['faithfulness'] for r in results_list) / len(results_list)
    avg_latency = sum(r['latency'] for r in results_list) / len(results_list)
    avg_precision = sum(r.get('precision', 0) for r in results_list) / len(results_list)
    total_cost = sum(r.get('cost', 0) for r in results_list)
    
    return {
        "avg_relevance": avg_relevance,
        "avg_faithfulness": avg_faithfulness,
        "avg_latency": avg_latency,
        "avg_precision": avg_precision,
        "total_cost": total_cost
    }


