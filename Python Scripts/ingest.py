from neo4j import GraphDatabase

# Connection details
URI = "neo4j://your-server-address:7687"
AUTH = ("your-username", "your-password")

# Query to load the CSV file
query_load_csv = """
LOAD CSV WITH HEADERS FROM 'file:///path/to/your/output_file.csv' AS row
CALL {
    WITH row
    MERGE (t:SubjectNode { name: row['Subject'] })
    MERGE (o:ObjectNode { name: COALESCE(row['Object'], 'None') })
    MERGE (t)-[r:RELATIONSHIP { name: toUpper(row['Relationship']) }]-(o)
} IN TRANSACTIONS;
"""

# Queries to verify the data
query_nodes = "MATCH (n) RETURN COUNT(n) AS TotalNodes;"
query_relationships = "MATCH ()-[r]->() RETURN COUNT(r) AS TotalRelationships;"

try:
    # Connect to Neo4j
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            print("Starting CSV data load into Neo4j...")
            session.run(query_load_csv)
            print("CSV data has been successfully loaded into Neo4j!")

            # Verify nodes
            result_nodes = session.run(query_nodes)
            for record in result_nodes:
                print("Total Nodes in the Database:", record["TotalNodes"])

            # Verify relationships
            result_relationships = session.run(query_relationships)
            for record in result_relationships:
                print("Total Relationships in the Database:", record["TotalRelationships"])

except Exception as e:
    print(f"An error occurred: {e}")
