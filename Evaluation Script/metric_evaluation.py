# -*- coding: utf-8 -*-
"""Metric_evaluation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WIz9kq5dj4m75npq27jCqWzheLkgUpZd
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def get_top_cosine_similarity(csv_path, target_nct, recommended_ncts, batch_size=5):
    """
    Optimized function to calculate cosine similarity between a target NCT and recommendations.
    Processes data in batches to reduce memory usage.
    """
    # Load CSV data in chunks to reduce memory usage
    df_chunks = pd.read_csv(csv_path, chunksize=10000)

    # Initialize model (load only once to save memory)
    model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)

    # Step 1: Extract keywords for the target NCT
    target_keywords = None
    for chunk in df_chunks:
        target_data = chunk[chunk['Subject'] == target_nct]
        if not target_data.empty:
            target_keywords = ' '.join(target_data['Object'].tolist())
            break

    if not target_keywords:
        print(f"No data found for target NCT: {target_nct}")
        return

    # Step 2: Generate embedding for the target NCT
    target_embedding = model.encode([target_keywords])

    # Step 3: Process recommendations in batches
    results = []
    for i in range(0, len(recommended_ncts), batch_size):
        batch_ncts = recommended_ncts[i:i + batch_size]

        # Extract keywords for the batch
        batch_keywords = []
        df_chunks = pd.read_csv(csv_path, chunksize=10000)  # Reset chunks
        for chunk in df_chunks:
            batch_data = chunk[chunk['Subject'].isin(batch_ncts)]
            if not batch_data.empty:
                for nct in batch_ncts:
                    keywords = ' '.join(batch_data[batch_data['Subject'] == nct]['Object'].tolist())
                    batch_keywords.append((nct, keywords))

        # Generate embeddings for the batch
        batch_texts = [keywords for _, keywords in batch_keywords]
        batch_embeddings = model.encode(batch_texts)

        # Calculate cosine similarity
        batch_similarities = cosine_similarity(target_embedding, batch_embeddings)[0]

        # Store results
        for (nct, _), score in zip(batch_keywords, batch_similarities):
            results.append((nct, round(score, 4)))

    # Sort results and take top 10
    results.sort(key=lambda x: x[1], reverse=True)
    top_10 = results[:10]

    # Calculate average cosine similarity
    avg_similarity = round(sum(score for _, score in top_10) / len(top_10), 4)

    # Add average similarity as a new row
    table_data = [[f"{i+1}.", nct, f"{score:.4f}"] for i, (nct, score) in enumerate(top_10)]
    table_data.append(["Average", "", f"{avg_similarity:.4f}"])  # Add average row

    # Print formatted results
    print(f"\n## NCT ID: {target_nct}")
    print("\nRecommendations:\n")
    print(tabulate(
        table_data,
        headers=["Rank", "Trial ID", "Cosine Similarity"],
        tablefmt="github"
    ))
    print("\n" + "="*50 + "\n")

# Example usage
csv_path = '/content/output_file.csv'

First example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT00385736',
    ['NCT03029143', 'NCT00408629', 'NCT01482884', 'NCT02289417',
     'NCT05731128', 'NCT00659802', 'NCT01620255', 'NCT00488631',
     'NCT02065557', 'NCT03221036']
)

Second example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT00386607',
    ['NCT00402103', 'NCT00923091', 'NCT00281580', 'NCT01456169',
     'NCT00841672', 'NCT00698646', 'NCT00435162', 'NCT01204398',
     'NCT00151775', 'NCT06174766']
)

# Third example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT03518073',
    ['NCT00762411', 'NCT00477659', 'NCT00428090', 'NCT02754830',
     'NCT00843518', 'NCT05310071', 'NCT01849055', 'NCT04994483',
     'NCT02091362', 'NCT02670083']
)

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

def get_top_cosine_similarity(csv_path, target_nct, recommended_ncts):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Get all unique NCTs (target + recommendations)
    all_ncts = [target_nct] + recommended_ncts

    # Create keyword dictionary {nct: combined_keywords}
    nct_keywords = {}
    for nct in all_ncts:
        keywords = df[df['Subject'] == nct]['Object'].tolist()
        if not keywords:
            print(f"No data found for NCT: {nct}")
            return
        nct_keywords[nct] = ' '.join(keywords)

    # Initialize model
    model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)

    # Generate embeddings for all NCTs
    texts = [nct_keywords[nct] for nct in all_ncts]
    embeddings = model.encode(texts)

    # Calculate cosine similarities
    target_embedding = embeddings[0].reshape(1, -1)
    recommendation_embeddings = embeddings[1:]

    similarities = cosine_similarity(target_embedding, recommendation_embeddings)[0]

    # Create results list
    results = []
    for nct, score in zip(recommended_ncts, similarities):
        results.append((nct, round(score, 4)))

    # Sort by similarity score (descending) and take top 10
    results.sort(key=lambda x: x[1], reverse=True)
    top_10 = results[:10]

    # Print formatted results
    print(f"\n## NCT ID: {target_nct}")
    print("\nRecommendations:\n")
    print(tabulate(
        [[f"{i+1}.", nct, f"{score:.4f}"] for i, (nct, score) in enumerate(top_10)],
        headers=["Rank", "Trial ID", "Cosine Similarity"],
        tablefmt="github"
    ))
    print("\n" + "="*50 + "\n")

# Example usage
csv_path = '/content/output_file.csv'

# First example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT00385736',
    ['NCT03029143', 'NCT00408629', 'NCT01482884', 'NCT02289417',
     'NCT05731128', 'NCT00659802', 'NCT01620255', 'NCT00488631',
     'NCT02065557', 'NCT03221036']
)

# Second example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT00386607',
    ['NCT00402103', 'NCT00923091', 'NCT00281580', 'NCT01456169',
     'NCT00841672', 'NCT00698646', 'NCT00435162', 'NCT01204398',
     'NCT00151775', 'NCT06174766']
)

# Third example comparison
get_top_cosine_similarity(
    csv_path,
    'NCT03518073',
    ['NCT00762411', 'NCT00477659', 'NCT00428090', 'NCT02754830',
     'NCT00843518', 'NCT05310071', 'NCT01849055', 'NCT04994483',
     'NCT02091362', 'NCT02670083']
)

