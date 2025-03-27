import sys
import pysolr
import pandas 
import logging

# Adapted from https://sease.io/2023/01/apache-solr-neural-search-tutorial.html

## Solr configuration.
SOLR_ADDRESS = 'http://localhost:8983/solr/mdsoar'
# Create a client instance.
solr = pysolr.Solr(SOLR_ADDRESS, always_commit=True)

logger = logging.getLogger('solr_index')

BATCH_SIZE = 100

def index_documents(embedding_filename):
    csv = pandas.read_csv(embedding_filename)
    documents = []
    for i, row in csv.iterrows():
        doc = {
            "id": i,
            "text": row['filename'],
            "vector": row['vector']
        }
        documents.append(doc)

        # For debugging
        if i == 0:
            logging.warning(documents)

        # To index batches of documents at a time.
        if i % BATCH_SIZE == 0 and i != 0:
            # How you'd index data to Solr.
            solr.add(documents)
            documents = []
            print("==== indexed {} documents ======".format(i))
        # To index the rest, when 'documents' list < BATCH_SIZE.
        # if documents:
            solr.add(documents)
        print("finished")

def main():
    embedding_filename = sys.argv[1]
    index_documents(embedding_filename)

if __name__ == "__main__":
    main()
