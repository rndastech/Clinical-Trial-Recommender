import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss  # Facebook AI Similarity Search
import csv
import logging

# Setup logging to print to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU Configuration
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def get_embeddings(texts, tokenizer, model, batch_size=256):
    """
    Generate GPU-accelerated embeddings for a list of texts.
    Processes texts in batches to optimize memory usage.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            batch_embeddings = outputs.hidden_states[-1].mean(dim=1)  # Shape: (batch_size, hidden_size)
            embeddings.append(batch_embeddings.cpu().numpy())
            logger.debug(f"Processed batch {(i // batch_size) + 1}")
    embeddings = np.vstack(embeddings)  # Shape: (num_texts, hidden_size)
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index for the given embeddings.
    """
    dimension = embeddings.shape[1]
    logger.info("Building FAISS index on CPU...")
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors.")
    return index

def find_similar_objects(embeddings, index, objects, similarity_threshold=0.8, top_k=100):
    """
    For each embedding, find similar objects with similarity above the threshold.
    """
    logger.info("Searching for similar objects using FAISS...")
    faiss.normalize_L2(embeddings)
    similarities, indices = index.search(embeddings, top_k)

    similar_objects_list = []
    for idx, (sim, ind) in enumerate(zip(similarities, indices)):
        similar = [objects[index_] for score, index_ in zip(sim, ind) if score >= similarity_threshold and index_ != idx]
        similar_objects_list.append(similar)
    logger.info("Similarity search completed.")
    return similar_objects_list

def serialize_list_with_double_quotes(lst):
    """Serialize a list of strings ensuring all elements are wrapped in double quotes."""
    return '[' + ', '.join(['"{}"'.format(s.replace('"', '\\"')) for s in lst]) + ']'

def main():
    try:
        # Configuration
        model_name = "NovaSearch/stella_en_1.5B_v5"
        csv_path = "Object_Value_Counts.csv"
        embeddings_output_path = "object_embeddings.npy"
        objects_output_path = "objects_list.npy"
        similarity_threshold = 0.8
        top_k = 100
        filtered_output_path = "filtered_results_with_similars.csv"
        cleaned_output_path = "cleaned_file.csv"
        merged_file_path = "merged.csv"
        final_output_path = "output_file.csv"

        # Load tokenizer and model
        logger.info("Loading tokenizer and model")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        logger.info("Tokenizer and model loaded successfully")

        # Read CSV and generate embeddings
        logger.info(f"Reading CSV from {csv_path}")
        data = pd.read_csv(csv_path)
        objects = data["Object"].astype(str).tolist()
        logger.info(f"Loaded {len(objects)} objects from CSV")

        logger.info("Generating embeddings for all objects")
        embeddings = get_embeddings(objects, tokenizer, model)
        np.save(embeddings_output_path, embeddings)
        np.save(objects_output_path, np.array(objects))
        logger.info("Embeddings and objects saved successfully")

        # Build FAISS index and find similar objects
        index = build_faiss_index(embeddings)
        similar_objects_list = find_similar_objects(embeddings, index, objects, similarity_threshold, top_k)

        # Serialize similar objects and save results
        data["Similar_Objects"] = [serialize_list_with_double_quotes(lst) for lst in similar_objects_list]
        data.to_csv(filtered_output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Filtered results saved to {filtered_output_path}")

        # Remove duplicates
        logger.info("Removing duplicate objects")
        data = pd.read_csv(filtered_output_path)
        data["Similar_Objects"] = data["Similar_Objects"].str.lower()
        unique_data = data[~data.apply(lambda row: any(row['Object'].lower() in obj for obj in eval(row['Similar_Objects'])), axis=1)]
        unique_data.to_csv(cleaned_output_path, index=False)
        logger.info(f"Cleaned data saved to {cleaned_output_path}")

        # Update merged file
        logger.info("Updating merged file")
        cleaned_data = pd.read_csv(cleaned_output_path)
        merged_data = pd.read_csv(merged_file_path)
        cleaned_data["Similar_Objects"] = cleaned_data["Similar_Objects"].apply(eval)
        mapping = {similar.lower(): row["Object"] for _, row in cleaned_data.iterrows() for similar in row["Similar_Objects"]}
        merged_data["Object"] = merged_data["Object"].fillna("None").apply(lambda obj: mapping[obj.lower()] if obj.lower() in mapping else obj)
        merged_data.to_csv(final_output_path, index=False)
        logger.info(f"Final output saved to {final_output_path}")

    except Exception as e:
        logger.exception("An error occurred")

if __name__ == "__main__":
    main()
