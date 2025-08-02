import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from src.config import (
    MILVUS_HOST, MILVUS_PORT, TEXT_COLLECTION_NAME, TABLE_COLLECTION_NAME,
    EMBEDDING_DIM, TEXT_OUTPUT_JSON, TABLES_OUTPUT_JSON
)

class MilvusManager:
    def __init__(self):
        print("🔌 Connecting to Milvus...")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ Connected to Milvus.")

        print("🤖 Loading sentence transformer model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded.")
        
        self.text_col = self._get_or_create_collection(TEXT_COLLECTION_NAME, "Text document chunks")
        self.table_col = self._get_or_create_collection(TABLE_COLLECTION_NAME, "Table document chunks")
        self._create_indexes_if_needed()

    def _get_or_create_collection(self, name, description):
        """Loads a collection or creates it if it doesn't exist."""
        if name not in list_collections():
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page_no", dtype=DataType.INT64),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, description=description)
            collection = Collection(name, schema)
            print(f"✅ Collection '{name}' created.")
        else:
            collection = Collection(name=name)
            print(f"📁 Collection '{name}' already exists.")
        return collection

    def _embed_and_insert(self, collection, data, data_type):
        """Embeds content and inserts it into a Milvus collection."""
        if not data: return
        
        embeddings, sources, pages, types, contents = [], [], [], [], []
        print(f"📦 Preparing '{data_type}' data for insertion...")

        for i, chunk in enumerate(tqdm(data, desc=f"Embedding {data_type}s")):
            content_str = json.dumps(chunk['content']) if isinstance(chunk['content'], (list, dict)) else chunk['content']
            if not content_str.strip(): continue

            metadata = chunk.get("metadata", {})
            source = metadata.get("source_document", "unknown")
            page_no = int(metadata.get("page_number", metadata.get("chunk_id", i)))

            emb = self.encoder.encode(content_str)
            embeddings.append(emb.tolist())
            sources.append(source)
            pages.append(page_no)
            types.append(data_type)
            contents.append(content_str)

        if not embeddings:
            print(f"⚠️ No valid '{data_type}' data found to insert.")
            return

        print(f"🚀 Inserting {len(embeddings)} '{data_type}' vectors into '{collection.name}'...")
        collection.insert([embeddings, sources, pages, types, contents])
        collection.flush()
        print(f"✅ Data successfully inserted and flushed into '{collection.name}'.")

    def ingest_data(self):
        """Loads processed data from JSON files and ingests into Milvus."""
        try:
            with open(TEXT_OUTPUT_JSON, "r", encoding="utf-8") as f:
                text_data = json.load(f)["text_chunks"]
            with open(TABLES_OUTPUT_JSON, "r", encoding="utf-8") as f:
                table_data = json.load(f)["tables"]
        except (FileNotFoundError, KeyError) as e:
            print(f"❌ Error loading data: {e}. Run processing first.")
            return

        self._embed_and_insert(self.text_col, text_data, "text")
        self._embed_and_insert(self.table_col, table_data, "table")
        self.print_status()

    def _create_indexes_if_needed(self):
        """Creates indexes for collections if they don't exist."""
        print("\\n🏗 Creating indexes for collections (if they don't exist)...")
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        for col in [self.text_col, self.table_col]:
            if not col.has_index():
                col.create_index(field_name="embedding", index_params=index_params)
                print(f"✅ Index created for '{col.name}'.")
            else:
                print(f"✅ Index already exists for '{col.name}'.")

    def retrieve(self, query, top_k=5):
        """Searches collections and returns re-ranked results."""
        query_vec = self.encoder.encode(query).tolist()
        self.text_col.load()
        self.table_col.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        text_results = self.text_col.search([query_vec], "embedding", search_params, limit=top_k, output_fields=["source", "page_no", "type", "content"])
        table_results = self.table_col.search([query_vec], "embedding", search_params, limit=top_k, output_fields=["source", "page_no", "type", "content"])
        
        combined = { (res.entity.get("source"), res.entity.get("page_no")): res for res_list in [text_results, table_results] for res in res_list[0] }
        return sorted(list(combined.values()), key=lambda x: x.distance)

    def rag_answer(self, query):
        """Performs the full RAG pipeline: retrieve, prompt, and generate."""
        retrieved_hits = self.retrieve(query, top_k=5)
        if not retrieved_hits:
            return "I could not find relevant information to answer your question.", []

        context_block = "\\n\\n---\\n\\n".join([hit.entity.get("content", "") for hit in retrieved_hits])
        prompt = (
            "You are an expert AI assistant. Use only the provided context to answer the user's question. "
            "If the context doesn't contain the answer, state that you cannot answer. "
            f"--- CONTEXT ---\\n{context_block}\\n\\n--- END CONTEXT ---\\n\\n"
            f"Question: {query}\\nAnswer:"
        )
        
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key: raise ValueError("MISTRAL_API_KEY not found in .env file.")
        
        client = Mistral(api_key=api_key)
        print("🤖 Generating answer with Mistral AI...")
        response = client.chat.complete(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        print("✅ Answer generated.")
        
        retrieved_chunks = [{
            "content": hit.entity.get("content", ""), "source": hit.entity.get("source", "unknown"),
            "page_no": hit.entity.get("page_no", 0), "similarity_score": 1 - hit.distance,
            "type": hit.entity.get("type", "unknown")} for hit in retrieved_hits]
        
        return response.choices[0].message.content.strip(), retrieved_chunks

    def print_status(self):
        """Prints the number of entities in each collection."""
        self.text_col.flush()
        self.table_col.flush()
        print("\\n" + "="*50)
        print("📊 Collection Status:")
        print(f"  - Entities in '{self.text_col.name}': {self.text_col.num_entities}")
        print(f"  - Entities in '{self.table_col.name}': {self.table_col.num_entities}")
        print("="*50)