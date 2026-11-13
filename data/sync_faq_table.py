#!/usr/bin/env python3
"""
Synchronize an explicitly selected FAQ file with Weaviate.
Suitable for manual execution or a privately configured scheduler.
Header/data offsets and column mapping are explicit operator settings.
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime


def _resolve_settings_env(value):
    """Resolve required environment references in already parsed configuration."""
    import os
    import re
    if isinstance(value, dict):
        return {key: _resolve_settings_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_settings_env(item) for item in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([A-Z_][A-Z0-9_]*)(?::(int|float|bool))?\}", value)
    if match is None:
        return value
    name, kind = match.groups()
    content = os.environ.get(name)
    if content is None or not content.strip():
        raise RuntimeError(f"Required configuration variable is missing: {name}")
    if kind == "int":
        return int(content)
    if kind == "float":
        return float(content)
    if kind == "bool":
        if content.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(f"Expected a boolean configuration variable: {name}")
        return content.lower() in {"true", "1"}
    return content



# Add a parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.weaviate_faq_loader import FAQLoader

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"faq_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='FAQ Sync with Weaviate')
    parser.add_argument('--csv', required=True, type=str, help='Path to CSV file with FAQ data')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--weaviate-url', type=str, help='URL Weaviate server')
    parser.add_argument('--weaviate-api-key', type=str, help='API Key')
    parser.add_argument('--openai-api-key', type=str, help='OpenAI key')
    parser.add_argument('--class-name', type=str, default='FAQ', help='Class name in Weaviate')
    parser.add_argument('--header-row', type=int, default=0, help='Zero-based input header row')
    parser.add_argument('--data-start-row', type=int, help='First data row; defaults to the row after the header')
    parser.add_argument('--column-map', help='Private JSON mapping of canonical fields to input headers')
    
    args = parser.parse_args()
    column_mapping = None
    if args.column_map:
        with open(args.column_map, encoding='utf-8') as mapping_file:
            column_mapping = json.load(mapping_file)
    
    # Checking the existence of a CSV file
    if not os.path.exists(args.csv):
        logger.error(f"The error: file {args.csv} Not found.")
        return 1
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = _resolve_settings_env(json.load(f))
            logger.info(f"The configuration is downloaded from {args.config}")
        except Exception as e:
            logger.error(f"Error in the configuration: {e}")
    
    # Priority: Command line arguments > configuration file > environment variables
    weaviate_url = args.weaviate_url or config.get('WEAVIATE_URL') or os.getenv('WEAVIATE_URL')
    weaviate_api_key = args.weaviate_api_key or config.get('WEAVIATE_API_KEY') or os.getenv('WEAVIATE_API_KEY')
    openai_api_key = args.openai_api_key or config.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
    class_name = args.class_name or config.get('FAQ_CLASS_NAME') or 'FAQ'
    
    # Checking the existence of mandatory parameters
    if not weaviate_url:
        logger.error('Error: Not specified URL of the Weaviate server')
        return 1
    if not weaviate_api_key:
        logger.error('Error: Not specified API key Weaviate')
        return 1
    if not openai_api_key:
        logger.error('Error: Not specified API key OpenAI')
        return 1
    
    logger.info(f"Start of FAQ Synchronization from {args.csv} in Weaviate{weaviate_url})")
    logger.info(f"Use the title line: {args.header_row+1}, data start line: {(args.data_start_row if args.data_start_row is not None else args.header_row + 1) + 1}")
    
    # Create and launch a downloader
    try:
        loader = FAQLoader(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key,
            openai_api_key=openai_api_key,
            csv_file_path=args.csv,
            class_name=class_name,
            header_row=args.header_row,
            data_start_row=args.data_start_row,
            column_mapping=column_mapping
        )
        
        await loader.run()
        logger.info('FAQ sync successfully completed')
        return 0
    
    except Exception as e:
        logger.error(f"Errors in synchronization FAQ: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 