import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Results provided by user
# {
#     "OpenAI": {
#         "avg_relevance": 9.4,
#         "avg_faithfulness": 9.4,
#         "avg_latency": 2.40,
#         "avg_precision": 0.6,
#         "total_cost": 0.001725
#     },
#     "Local": {
#         "avg_relevance": 8.2,
#         "avg_faithfulness": 8.2,
#         "avg_latency": 326.34,
#         "avg_precision": 0.38,
#         "total_cost": 0.0
#     },
#     "PageIndex": {
#         "avg_relevance": 7.0,
#         "avg_faithfulness": 4.8,
#         "avg_latency": 7.15,
#         "avg_precision": 0.2,
#         "total_cost": 0.0
#     }
# }

def generate_charts():
    labels = ['OpenAI', 'Local', 'PageIndex']
    relevance = [9.4, 8.2, 7.0]
    faithfulness = [9.4, 8.2, 4.8]
    precision = [0.6 * 10, 0.38 * 10, 0.2 * 10] # Scaled to 0-10 for chart
    latency = [2.40, 326.34, 7.15]

    x = np.arange(len(labels))
    width = 0.25

    # Chart 1: Quality Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))
    rects1 = ax1.bar(x - width, relevance, width, label='Relevance', color='#4CAF50')
    rects2 = ax1.bar(x, faithfulness, width, label='Faithfulness', color='#2196F3')
    rects3 = ax1.bar(x + width, precision, width, label='Precision (x10)', color='#FFC107')

    ax1.set_ylabel('Score (0-10)')
    ax1.set_title('RAG Comparison: Quality Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_ylim(0, 10.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig('evaluation/results/quality_comparison.png')
    print("Saved quality_comparison.png")

    # Chart 2: Latency (Log Scale)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rects_lat = ax2.bar(labels, latency, color=['#4CAF50', '#F44336', '#FF9800'])
    ax2.set_ylabel('Latency (seconds)')
    ax2.set_title('RAG Comparison: Latency (Log Scale)')
    ax2.set_yscale('log')
    
    for i, v in enumerate(latency):
        ax2.text(i, v * 1.1, f"{v:.1f}s", ha='center')

    plt.tight_layout()
    plt.savefig('evaluation/results/latency_comparison.png')
    print("Saved latency_comparison.png")

if __name__ == "__main__":
    if not os.path.exists('evaluation/results'):
        os.makedirs('evaluation/results')
    generate_charts()
