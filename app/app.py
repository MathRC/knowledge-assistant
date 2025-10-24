"""
app.py

Streamlit web application for a RAG-based knowledge assistant that answers queries
about the European Commission's English Style Guide.

Connects to Weaviate for retrieval, uses BAAI/bge-large-en-v1.5 for embeddings,
and Claude 3 Sonnet (via Amazon Bedrock) for response generation.
"""

import streamlit as st
import yaml
import weaviate
import weaviate.classes.query as wq
from sentence_transformers import SentenceTransformer
import torch
import boto3
import json
import os
import time
from botocore.exceptions import ClientError
from urllib.parse import urlparse


def load_config():
    """Loads configuration from config.yaml."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("❌ Configuration file 'config.yaml' not found.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"❌ Error parsing config.yaml: {e}")
        st.stop()


# Load configuration
config = load_config()

# Extract key settings
AWS_REGION = config["llm"]["aws_region"]
CLAUDE_MODEL_ID = config["llm"]["model_id"]
WEAVIATE_URL = config["weaviate"]["url"]
COLLECTION_NAME = config["weaviate"]["collection_name"]
EMBEDDING_MODEL_NAME = config["embedding_model"]


# Prompt template enforcing evidence-based responses
ENGINEERED_PROMPT_TEMPLATE = """
You are a specialist assistant for the European Commission's English Style Guide.

STRICT GUIDELINES:
1. Answer ONLY using information explicitly stated in the provided context.
2. Quote directly from the context using quotation marks when possible.
3. If the context lacks information to answer the question, respond: "The European Commission's English Style Guide does not address this topic in the provided sections."
4. Never use general knowledge about style guides, EU practices, or writing conventions.
5. Be precise about what the guide says vs. what it doesn't say.

Context from the Style Guide:
---
{context}
---

Question: {question}

Response:
"""


@st.cache_resource
def load_models_and_clients():
    """
    Initializes and caches embedding model, Bedrock client, and Weaviate collection.
    Ensures services are ready before returning.
    """
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        st.error(f"❌ Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        st.stop()

    # Initialize AWS Bedrock client
    try:
        bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        st.error(f"❌ Failed to initialize Bedrock client: {e}")
        st.stop()

    # Parse Weaviate URL
    try:
        parsed = urlparse(WEAVIATE_URL)
        host = parsed.hostname
        port = parsed.port or 8080
    except Exception as e:
        st.error(f"❌ Invalid WEAVIATE_URL: {WEAVIATE_URL}")
        st.stop()

    # Connect to Weaviate
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=False,
        )

        # Wait for readiness (required for weaviate-client==4.16.9)
        for _ in range(60):
            try:
                if client.is_ready():
                    break
            except:
                time.sleep(1)
        else:
            raise RuntimeError("Weaviate did not become ready within 60 seconds")

        collection = client.collections.get(COLLECTION_NAME)
        return embedding_model, bedrock_runtime, collection

    except Exception as e:
        st.error(f"❌ Could not connect to Weaviate at {WEAVIATE_URL}. Error: {e}")
        st.info("💡 Make sure:")
        st.markdown("- Weaviate container is healthy")
        st.markdown("- Collection `'StyleGuide'` was created by `ingest_data.py`")
        st.stop()


# === BACKEND FUNCTIONS ===

def get_retrieved_context(question: str, collection, embedding_model, limit: int = 5) -> str:
    """
    Retrieves relevant context using hybrid search with section-based filtering.
    """
    query_vector = embedding_model.encode(question).tolist()
    response = collection.query.hybrid(
        query=question,
        vector=query_vector,
        alpha=0.7,
        limit=limit,
        filters=wq.Filter.by_property("method").like("section_based*"),
        return_properties=["text"]
    )
    if not response.objects:
        return ""
    return "\n\n---\n\n".join([obj.properties["text"] for obj in response.objects])


def get_rag_response(question: str, context: str, bedrock_runtime) -> dict:
    """
    Generates a grounded response using Claude via Amazon Bedrock.
    """
    if not context:
        return {
            "answer": "The European Commission's English Style Guide does not address this topic in the provided sections.",
            "context": "",
        }

    final_prompt = ENGINEERED_PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
        ],
    }
    body = json.dumps(payload)

    try:
        response = bedrock_runtime.invoke_model(body=body, modelId=CLAUDE_MODEL_ID)
        response_body = json.loads(response["body"].read())
        answer = response_body["content"][0]["text"]
        return {"answer": answer, "context": context}
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "AccessDeniedException":
            msg = "Access to Claude denied. Check AWS IAM permissions."
        elif error_code == "ThrottlingException":
            msg = "Request rate limit reached. Please wait."
        else:
            msg = f"AWS Error: {e.response['Error']['Message']}"
        return {"answer": f"Error: {msg}", "context": context}
    except Exception as e:
        return {"answer": f"Unexpected error: {str(e)}", "context": context}


# === Initialize Models and Clients ===
embedding_model, bedrock_runtime, collection = load_models_and_clients()

# === Initialize Chat History ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Sidebar ===
with st.sidebar:
    st.header("About")
    st.markdown("""
    A RAG system for translators querying the EU English Style Guide.

    Powered by:
    - BAAI/bge-large-en-v1.5
    - Weaviate
    - Claude 3 Sonnet (Bedrock)
    """)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# === Main Page ===
st.title("📝 EU Style Guide Assistant")
st.write("Ask a question about the European Commission's English Style Guide.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("e.g., How should I format dates and times?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()

            # Step 1: Retrieve context
            retrieval_start = time.time()
            context = get_retrieved_context(prompt, collection, embedding_model)
            retrieval_end = time.time()

            # Step 2: Generate RAG response
            llm_start = time.time()
            result = get_rag_response(prompt, context, bedrock_runtime)
            llm_end = time.time()

            total_time = time.time() - start_time

            # Display answer
            st.markdown(result["answer"])

            # Show context and metrics
            with st.expander("Show Retrieved Context & Performance"):
                st.text(result["context"])
                st.markdown("---")
                st.markdown("**Performance Metrics:**")
                st.write(f"- Retrieval Time: {retrieval_end - retrieval_start:.2f}s")
                st.write(f"- LLM Time: {llm_end - llm_start:.2f}s")
                st.write(f"- Total: {total_time:.2f}s")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})