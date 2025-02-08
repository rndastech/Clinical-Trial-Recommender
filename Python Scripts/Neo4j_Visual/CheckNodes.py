from neo4j import GraphDatabase

# Connection details
URI = "neo4j://your-server-address:7687"
AUTH = ("your-username", "your-password")

# Queries to count nodes and relationships
query_nodes = "MATCH (n) RETURN COUNT(n) AS TotalNodes;"
query_relationships = "MATCH ()-[r]->() RETURN COUNT(r) AS TotalRelationships;"

def count_nodes_and_relationships():
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                # Count nodes
                result_nodes = session.run(query_nodes)
                for record in result_nodes:
                    print("Total Nodes in the Database:", record["TotalNodes"])

                # Count relationships
                result_relationships = session.run(query_relationships)
                for record in result_relationships:
                    print("Total Relationships in the Database:", record["TotalRelationships"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    count_nodes_and_relationships()
