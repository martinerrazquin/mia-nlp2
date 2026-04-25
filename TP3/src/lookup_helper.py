from pinecone.db_data.index import Index

class PineconeLookup:
    def __init__(self, index: Index, namespace: str, top_k: int):
        """
        Initializes the PineconeLookup helper for performing semantic searches.

        Args:
            index (Index): Pinecone index instance to query against.
            namespace (str): Namespace within the index to scope the search.
            top_k (int): Number of top matching results to retrieve.
        """
        self.index = index
        self.namespace = namespace
        self.top_k = top_k

    def lookup(self, query: str) -> str:
        """
        Performs a semantic search query against the Pinecone index and returns
        the concatenated text results.

        The method queries the index using the provided text input and retrieves
        the top_k most relevant matches, extracting their "text" fields.

        Args:
            query (str): Input query string for semantic search.

        Returns:
            str: A single string containing the joined text results, separated
                by double newlines.
        """
        # look query up the known index
        results = self.index.search(
            namespace=self.namespace,
            query={
                "inputs": {"text": query}, 
                "top_k": self.top_k
            }, # type: ignore
            fields=["text"]
        )

        # parse -> join as str
        return "\n\n".join(
            result['fields']['text']
            for result in results['result']['hits']
        )
