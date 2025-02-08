import os
import pandas as pd
from groq import Groq
import csv

# Configure Groq client
api_key = "your-api-key"
client = Groq(api_key=api_key)

# File paths
input_file_path = "<path-to-input-file>"
output_file_path = "<path-to-unrefined-output-file>"

# Load and prepare data
df = pd.read_csv(input_file_path, low_memory=False).dropna(how="all")

# Merge relevant columns
columns_to_merge = ["Study Title", "Primary Outcome Measures", 
                   "Secondary Outcome Measures", "criteria"]
df["Merged_Content"] = df[columns_to_merge].apply(
    lambda row: " \n".join(row.values.astype(str)), axis=1
)

# Improved prompt template
PROMPT_TEMPLATE = (
    "You are a clinical trial data expert. Extract relationships STRICTLY in this format:\n"
    "RELATIONSHIP[TAB]OBJECT\n\n"
    "Relationships to extract:\n"
    "- involves: Disease/condition name\n"
    "- evaluates: Drug/intervention name\n"
    "- measures_primary: Primary outcome (≤5 words)\n"
    "- measures_secondary: Secondary outcome (≤5 words)\n"
    "- has_criteria: Eligibility criteria (≤5 words)\n\n"
    "Rules:\n"
    "1. OBJECT must be ONLY the extracted value - no labels, quotes, or prefixes\n"
    "2. Use exact medical terminology from the text\n"
    "3. Skip relationships if information is missing\n"
    "4. Use TAB separator between relationship and object\n\n"
    "Example output:\n"
    "involves\tAlzheimer's Disease\n"
    "evaluates\tIntravenous Sabirnetug\n\n"
    "Process this clinical trial data:\n{content}"
)

def extract_relationships(content):
    """Process content and return list of (relationship, object) tuples"""
    if pd.isnull(content) or not content.strip():
        return []

    # Truncate long content
    content = str(content)
    if len(content) > 2000:
        content = content[:2000] + "... [TRUNCATED]"

    try:
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(content=content)
            }],
            temperature=0.3,
            max_tokens=1024
        )
        response = completion.choices[0].message.content
        
        # Parse response lines
        relationships = []
        for line in response.splitlines():
            if "\t" in line:
                rel, obj = line.split("\t", 1)
                rel = rel.strip().lower()
                obj = obj.strip(" '")  # Clean quotes
                
                # Validate relationships
                if rel in {'involves', 'evaluates', 'measures_primary', 
                          'measures_secondary', 'has_criteria'} and obj:
                    relationships.append((rel, obj))
        
        return relationships
    
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return []

# Initialize output file
with open(output_file_path, 'w') as f:
    f.write("Subject,Relationship,Object\n")

# Process rows and write results
for index, row in df.iterrows():
    subject_id = row.get("NCT Number", f"ROW_{index}")
    relationships = extract_relationships(row["Merged_Content"])
    
    if relationships:
        with open(output_file_path, 'a') as f:
            for rel, obj in relationships:
                f.write(f"{subject_id},{rel},{obj}\n")
    
    print(f"Processed row {index} - Extracted {len(relationships)} relationships")

print(f"Processing complete. Results saved to: {output_file_path}")

# Define the input and output file paths
input_file = output_file_path
output_file = '<path-to-output-file>'  # Save to a different file to avoid overwriting prematurely

# Process the CSV to keep only the first three columns
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Check if the row has at least three columns before slicing
        if len(row) >= 3:
            writer.writerow(row[:3])

# Read the updated CSV file
try:
    df = pd.read_csv(output_file)

    # Get the value counts for the 'Object' column
    object_value_counts = df['Object'].value_counts()

    # Display the value counts
    print(object_value_counts)

    # Save the value counts to a CSV file
    object_value_counts.to_csv('Object_Value_Counts.csv', header=['Count'])
except pd.errors.EmptyDataError:
    print("The cleaned CSV file is empty or invalid. Please check the input file.")
