import asyncio
import json
import os
import ssl
from typing import Dict, List

import aiohttp
import certifi

from .fk_dataset import FKDataset

async def _call_gpt_async(
    session: aiohttp.ClientSession, 
    model_name: str, 
    query: Dict[str, str]
):
  try:
      payload = {
          "model": model_name,
          "messages": query,
          "response_format": {
              "type": "json_schema",
              "json_schema": {
                  "name": "company_info",
                  "strict": True,
                  "schema": FKDataset.model_json_schema(),
              },
          },
      }
      async with session.post(
          url="https://api.openai.com/v1/chat/completions",
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
          },
          json=payload,
          ssl=ssl.create_default_context(cafile=certifi.where()),
      ) as response:
          print(response)
          response = await response.json()
      if "error" in response:
          print(
              f"OpenAI request failed with error {response['error']}"
          )
      return json.loads(response["choices"][0]["message"]["content"])
  except Exception as e:
      print(f"Request failed: {e}")

async def invoke_openai(
    model_name: str, 
    query: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            _call_gpt_async(session, model_name, query)
        ]
        generations = await asyncio.gather(*tasks)

    return [gen for gen in generations if gen is not None]