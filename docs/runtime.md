# Runtime and configuration

Docker Compose defines Weaviate, QnA and reranker services. Review the runtime requirements and supply local Confluence/Weaviate/provider settings before connecting an approved corpus. Run ingestion as a module from the root with `python -m integrations.confluence_api.weaviate_upload`. Document and FAQ inputs must be supplied separately by their authorized owner.

Use the component navigation and validation boundaries in the root [README](../README.md). Runtime services and external providers are not started by the offline replay.
