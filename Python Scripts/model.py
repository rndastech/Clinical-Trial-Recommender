from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Connection details
URI = "neo4j://your-server-address:7687"
AUTH = ("your-username", "your-password")

# Function to project the graph
def project_graph(tx):
    query = """
    CALL gds.graph.project(
        'clinicalTrialsGraph',
        ['SubjectNode', 'ObjectNode'],
        {
            RELATIONSHIP: {orientation: 'UNDIRECTED'}
        }
    )
    """
    tx.run(query)
    print("Graph successfully projected with undirected relationships.")

# Function to check if the graph is already projected
def is_graph_projected(tx):
    query = """
    CALL gds.graph.exists('clinicalTrialsGraph')
    YIELD exists
    RETURN exists
    """
    result = tx.run(query)
    return result.single()["exists"]

# Function to drop and re-project the graph
def reproject_graph(tx):
    query = """
    CALL gds.graph.drop('clinicalTrialsGraph', false)
    """
    tx.run(query)
    print("Dropped existing graph projection.")
    project_graph(tx)
    print("Graph re-projected successfully.")

# Function to check if a node exists in the graph
def check_node_exists(tx, node_name, label):
    query = f"""
    MATCH (n:{label} {{name: TRIM($node_name)}})
    RETURN COUNT(n) > 0 AS exists
    """
    print(f"Checking existence of node: {node_name.strip()} with label: {label}")
    result = tx.run(query, node_name=node_name.strip())
    exists = result.single()["exists"]
    print(f"Node exists: {exists}")
    return exists

# Function to find similar trials using GDS node similarity
def find_similar_trials(tx, trial_id):
    query = f"""
    CALL gds.nodeSimilarity.stream('clinicalTrialsGraph')
    YIELD node1, node2, similarity
    WHERE gds.util.asNode(node1).name = $trial_id
      AND gds.util.asNode(node1):SubjectNode
      AND gds.util.asNode(node2):SubjectNode
    RETURN gds.util.asNode(node2).name AS similarTrial, similarity
    ORDER BY similarity DESC
    LIMIT 10
    """
    print(f"Executing similarity query for trial ID: {trial_id}")
    result = tx.run(query, trial_id=trial_id)
    results = [{"trial": record["similarTrial"], "similarity": record["similarity"]} for record in result]
    print(f"Results: {results}")
    return results

# Main script to query for similar trials
def main():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        with driver.session() as session:
            # Check if the graph is already projected
            graph_exists = session.read_transaction(is_graph_projected)

            if graph_exists:
                print("Graph already exists. Re-projecting...")
                reproject_graph(session)
            else:
                print("Projecting the graph for the first time...")
                session.write_transaction(project_graph)

            # Get input from user
            trial_id = input("Enter the trial ID (e.g., NCT00752622): ")

            # Check if the node exists
            node_exists = session.read_transaction(check_node_exists, trial_id, 'SubjectNode')
            if not node_exists:
                print(f"Node '{trial_id}' does not exist in the graph. Please enter a valid trial ID.")
                return

            # Find similar trials
            print(f"Finding trials similar to {trial_id}...")
            similar_trials = session.read_transaction(find_similar_trials, trial_id)

            if similar_trials:
                print("Top 10 similar trials:")
                for i, trial in enumerate(similar_trials, 1):
                    print(f"{i}. Trial ID: {trial['trial']}, Similarity: {trial['similarity']:.4f}")
            else:
                print("No similar trials found.")
    except ServiceUnavailable as e:
        print(f"Failed to connect to Neo4j: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
