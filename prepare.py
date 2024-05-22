import os
from sqlalchemy import URL
from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.readers.web import SimpleWebPageReader
from dotenv import load_dotenv

load_dotenv()
tidb_username = os.environ['TIDB_USERNAME']
tidb_password = os.environ['TIDB_PASSWORD']
tidb_host = os.environ['TIDB_HOSTNAME']
tidb_database = os.environ['TIDB_DATABASE_NAME']

# ------------------------------------------------------------------------
# LlamaIndex - Amazon Bedrock
llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")
embed_model = BedrockEmbedding(model="amazon.titan-embed-text-v1")

Settings.llm = llm
Settings.embed_model = embed_model

def main():
    tidb_connection_url = URL(
        "mysql+pymysql",
        username=tidb_username,
        password=tidb_password,
        host=tidb_host,
        port=4000,
        database=tidb_database,
        query={"ssl_verify_cert": True, "ssl_verify_identity": True},
    )

    tidbvec = TiDBVectorStore(
        connection_string=tidb_connection_url,
        table_name="llama_index_rag",
        distance_strategy="cosine",
        vector_dimension=1536,
        drop_existing_table=False,
    )

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        [
            "https://docs.pingcap.com/ja/tidb/stable/overview",
            "https://docs.pingcap.com/ja/tidb/stable/release-7.5.0",
            "https://docs.pingcap.com/ja/tidb/stable/mysql-compatibility",
            "https://docs.pingcap.com/ja/tidbcloud/tidb-cloud-intro#architecture"
        ]
    )

    storage_context = StorageContext.from_defaults(vector_store=tidbvec)
    tidb_vec_index = VectorStoreIndex.from_vector_store(tidbvec)
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    
    return tidb_vec_index

main()