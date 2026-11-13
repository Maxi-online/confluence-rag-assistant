"""Use the actual ingestion cleaner on synthetic Confluence-like HTML."""
import json
from integrations.confluence_api.text_utils import static_clean_html

def run():
    html="<h1>Demo handbook</h1><script>private_script()</script><p>Support hours: 09:00–17:00.</p><ul><li>Use the support portal.</li></ul>"
    return {"mode":"synthetic-ingestion-cleaner","page_id":"demo-handbook","text":static_clean_html(html),"embeddings_or_llm_executed":False,"weaviate_acl_verified":False}
if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
