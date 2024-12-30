import asyncio
import os
from devtools import debug
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext


logfire.configure(send_to_logfire='if-token-present')

@dataclass
class Deps:
    client: AsyncClient


agent = Agent(
    'ollama:qwen2',
    # 'ollama:llama3.2',
    # 'openai:gpt-4o-mini',
    system_prompt= (
        'You are an personal assistant to the user.'
        'You are able to answer questions and help the user with their tasks.'
        'use the `get_current_weather` tool to get the current weather in a given location.'
        'use the `get_current_datetime` tool to get the current date and time in UTC.'
    ),
    deps_type=Deps,
    retries=2
)


@agent.tool
async def get_current_datetime(ctx: RunContext) -> str:
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


@agent.tool
async def get_current_weather(ctx: RunContext, location: str) -> str:
    return f'The weather in {location} is sunny and warm.'


async def main():
    deps = Deps(client=AsyncClient())
    async with agent.run_stream("Please write a python script for pandas dataframe ploting", deps=deps) as result:
        async for message in result.stream_text(delta=True):
            print(message)

    # result = await agent.run("Please write a python script for pandas dataframe ploting", deps=deps)
    # debug(result)
    # print(result.data)

asyncio.run(main())