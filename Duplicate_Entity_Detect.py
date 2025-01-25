#First Execute This

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
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
            # Use the mean of the last hidden state as the embedding
            batch_embeddings = outputs.hidden_states[-1].mean(dim=1)  # Shape: (batch_size, hidden_size)
            embeddings.append(batch_embeddings.cpu().numpy())
            logger.debug(f"Processed batch {(i // batch_size) + 1}")
    embeddings = np.vstack(embeddings)  # Shape: (num_texts, hidden_size)
    return embeddings

def main():
    try:
        # Load tokenizer and model with GPU support
        logger.info("Loading tokenizer and model")
        model_name = "NovaSearch/stella_en_1.5B_v5"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        logger.info("Tokenizer and model loaded successfully")

        # Configuration
        csv_path = "/kaggle/input/objectcount2/Object_Value_Counts2.csv"  # Update this path as needed
        embeddings_output_path = "object_embeddings.npy"
        objects_output_path = "objects_list.npy"
        embedding_batch_size = 256  # Adjust based on GPU memory

        # Read CSV
        logger.info(f"Reading CSV from {csv_path}")
        data = pd.read_csv(csv_path)
        if "Object" not in data.columns:
            logger.error("'Object' column missing in CSV")
            raise ValueError("'Object' column missing in CSV")
        objects = data["Object"].astype(str).tolist()  # Ensure all objects are strings
        logger.info(f"Loaded {len(objects)} objects from CSV")

        # Generate embeddings for all objects
        logger.info("Generating embeddings for all objects")
        embeddings = get_embeddings(objects, tokenizer, model, batch_size=embedding_batch_size)
        logger.info("Embeddings generated successfully")

        # Save embeddings and objects
        logger.info(f"Saving embeddings to {embeddings_output_path}")
        np.save(embeddings_output_path, embeddings)
        logger.info(f"Saving objects list to {objects_output_path}")
        np.save(objects_output_path, np.array(objects))
        logger.info("Embeddings and objects saved successfully")

    except Exception as e:
        logger.exception("An error occurred during embedding computation")

if __name__ == "__main__":
    main()

# Then Execute This
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss  # Facebook AI Similarity Search
import csv  # Import the csv module for quoting constants

# GPU Configuration
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def build_faiss_index(embeddings, use_gpu=False):
    """
    Build a FAISS index for the given embeddings.
    """
    try:
        dimension = embeddings.shape[1]
        print("Building FAISS index on CPU...")
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity if vectors are normalized)

        # Skip GPU indexing to prevent OOM
        # Uncomment the following lines if you resolve the OOM issue and want to use GPU FAISS
        # if use_gpu and faiss.get_num_gpus() > 0:
        #     print("Moving FAISS index to GPU...")
        #     res = faiss.StandardGpuResources()
        #     index = faiss.index_cpu_to_gpu(res, 0, index)

        # Normalize embeddings to unit length for cosine similarity
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        print(f"FAISS index built with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print("Error in build_faiss_index:", e)
        raise

def find_similar_objects(embeddings, index, objects, similarity_threshold=0.7, top_k=100):
    """
    For each embedding, find similar objects with similarity above the threshold.
    """
    try:
        print("Searching for similar objects using FAISS...")
        faiss.normalize_L2(embeddings)  # Ensure embeddings are normalized

        # Perform similarity search
        # FAISS searches for top_k similar vectors; we'll filter them based on the threshold
        similarities, indices = index.search(embeddings, top_k)

        similar_objects_list = []
        total_objects = len(objects)
        for idx, (sim, ind) in enumerate(zip(similarities, indices)):
            # Filter out self-match and apply the similarity threshold
            similar = []
            for score, index_ in zip(sim, ind):
                if index_ == idx:
                    continue  # Skip self
                if score < similarity_threshold:
                    continue
                similar.append(objects[index_])
            similar_objects_list.append(similar)

            # Print progress every 1000 objects
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_objects:
                print(f"Processed {idx + 1} / {total_objects} objects.")

        print("Similarity search completed.")
        return similar_objects_list
    except Exception as e:
        print("Error in find_similar_objects:", e)
        raise

def serialize_list_with_double_quotes(lst):
    """
    Serialize a list of strings ensuring all elements are wrapped in double quotes.

    Args:
        lst (list of str): The list to serialize.

    Returns:
        str: The serialized list as a string with all elements in double quotes.
    """
    # Escape any existing double quotes in the strings
    escaped_lst = [s.replace('"', '\\"') for s in lst]
    # Wrap each string with double quotes
    quoted_lst = ['"{}"'.format(s) for s in escaped_lst]
    # Join into a list-like string
    return '[' + ', '.join(quoted_lst) + ']'

def main():
    try:
        # Clear GPU cache (if using GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Configuration
        embeddings_input_path = "object_embeddings.npy"
        objects_input_path = "objects_list.npy"
        csv_path = "/kaggle/input/objectcount2/Object_Value_Counts2.csv"  # Update this path as needed
        output_path = "filtered_results_with_similars.csv"
        similarity_threshold = 0.8
        top_k = 100  # Maximum number of similar objects to retrieve per object

        # Load embeddings and objects
        print(f"Loading embeddings from {embeddings_input_path}...")
        embeddings = np.load(embeddings_input_path)
        print(f"Loading objects list from {objects_input_path}...")
        objects = np.load(objects_input_path).tolist()
        print(f"Loaded {len(objects)} objects and their embeddings.")

        # Build FAISS index
        index = build_faiss_index(embeddings, use_gpu=False)

        # Find similar objects
        similar_objects_list = find_similar_objects(
            embeddings, index, objects, similarity_threshold=similarity_threshold, top_k=top_k
        )

        # Read original CSV
        print(f"Reading CSV from {csv_path}...")
        data = pd.read_csv(csv_path)
        if "Object" not in data.columns:
            print("Error: 'Object' column missing in CSV.")
            raise ValueError("'Object' column missing in CSV")
        print("CSV loaded successfully.")

        # Serialize the similar objects lists with double quotes
        print("Serializing similar objects with consistent double quotes...")
        serialized_similar_objects = [serialize_list_with_double_quotes(lst) for lst in similar_objects_list]

        # Add the serialized similar objects as a new column
        print("Adding similar objects to the DataFrame...")
        data["Similar_Objects"] = serialized_similar_objects

        # Save the final DataFrame to CSV with proper quoting
        print(f"Saving the final DataFrame to {output_path}...")
        data.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved filtered results with similar objects to {output_path}.")

    except Exception as e:
        print("An error occurred during similarity search and CSV augmentation:", e)

if __name__ == "__main__":
    main()

