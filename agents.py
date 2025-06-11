"""
OpenAI Agents Framework
Core components for running AI agents with tools and tracing
"""

from __future__ import annotations

import asyncio
import uuid
import time
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Sequence

from openai import AsyncOpenAI
from pydantic import BaseModel
import os

# Type definitions
T = TypeVar('T', bound=BaseModel)

class ModelSettings:
    """Model configuration settings"""
    def __init__(self, tool_choice: str = "auto"):
        self.tool_choice = tool_choice

class WebSearchTool:
    """Web search tool for agents"""
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for current information about locations, landmarks, and topics"
    
    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

@dataclass
class RunResult:
    """Result from running an agent"""
    response: Any
    
    def final_output_as(self, output_type: Type[T]) -> T:
        """Extract the final output as a specific type"""
        if isinstance(self.response, output_type):
            return self.response
        
        # If response is a string, try to parse it as the output type
        if isinstance(self.response, str):
            try:
                # Try to parse as JSON first
                data = json.loads(self.response)
                return output_type(**data)
            except:
                # If not JSON, create with the string as output
                if hasattr(output_type, 'output'):
                    return output_type(output=self.response)
                else:
                    # Try to create with first field
                    fields = list(output_type.__fields__.keys())
                    if fields:
                        return output_type(**{fields[0]: self.response})
        
        # Fallback: return the response as-is
        return self.response

class Agent:
    """AI Agent with tools and model settings"""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o-mini",
        tools: Optional[List[WebSearchTool]] = None,
        model_settings: Optional[ModelSettings] = None,
        output_type: Optional[Type[BaseModel]] = None
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.model_settings = model_settings or ModelSettings()
        self.output_type = output_type
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def run(self, user_input: str) -> Any:
        """Run the agent with user input"""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_input}
        ]
        
        # Prepare tools for OpenAI API
        tools = None
        if self.tools:
            tools = [tool.to_dict() for tool in self.tools]
        
        # Make API call
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=self.model_settings.tool_choice if tools else None,
                temperature=0.3
            )
            
            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                # For now, simulate tool responses
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "web_search":
                        # Simulate web search result
                        search_query = json.loads(tool_call.function.arguments)["query"]
                        search_result = f"Search results for '{search_query}': Current information about the location including recent updates, popular attractions, and relevant details."
                        
                        # Add tool response to conversation
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.dict()]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_result
                        })
                
                # Get final response after tool use
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )
                
                content = final_response.choices[0].message.content
            else:
                content = message.content
            
            # Parse output if output_type is specified
            if self.output_type and content:
                try:
                    # Try to parse as JSON first
                    data = json.loads(content)
                    return self.output_type(**data)
                except:
                    # If not JSON, create with content as output
                    if hasattr(self.output_type, 'output'):
                        return self.output_type(output=content)
                    else:
                        # Try to create with first field
                        fields = list(self.output_type.__fields__.keys())
                        if fields:
                            return self.output_type(**{fields[0]: content})
            
            return content
            
        except Exception as e:
            print(f"Error running agent {self.name}: {str(e)}")
            # Return a fallback response
            if self.output_type:
                if hasattr(self.output_type, 'output'):
                    return self.output_type(output=f"Error generating content for {self.name}")
                else:
                    fields = list(self.output_type.__fields__.keys())
                    if fields:
                        return self.output_type(**{fields[0]: f"Error generating content for {self.name}"})
            return f"Error generating content for {self.name}"

class Runner:
    """Agent runner utility"""
    
    @staticmethod
    async def run(agent: Agent, user_input: str) -> RunResult:
        """Run an agent and return results"""
        response = await agent.run(user_input)
        return RunResult(response=response)

# Tracing utilities
def gen_trace_id() -> str:
    """Generate a unique trace ID"""
    return str(uuid.uuid4())[:8]

@contextmanager
def trace(name: str, trace_id: str):
    """Context manager for tracing"""
    print(f"Starting trace: {name} ({trace_id})")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Completed trace: {name} ({trace_id}) in {end_time - start_time:.2f}s")

@contextmanager 
def custom_span(name: str):
    """Custom span for detailed tracing"""
    print(f"Starting span: {name}")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Completed span: {name} in {end_time - start_time:.2f}s")