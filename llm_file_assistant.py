"""CLI assistant that routes user requests to file tools with optional OpenAI tool calling."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from fs_tools import list_files, read_file, search_in_file, write_file


ToolResult = Union[Dict[str, Any], List[Dict[str, Any]]]
ToolFunction = Callable[..., ToolResult]


TOOLS: dict[str, ToolFunction] = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a TXT, PDF, or DOCX resume file and extract its text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Absolute or relative path to a supported file.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files inside a directory, optionally filtered by extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "extension": {
                        "type": ["string", "null"],
                        "description": "Optional extension filter such as .pdf or .txt.",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write plain text content to a file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for a keyword inside a supported file and return matching snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "keyword": {"type": "string"},
                },
                "required": ["filepath", "keyword"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a resume file assistant.
Use the available tools to answer questions about resume files.
Prefer tool calls over guessing. When summarizing results, be concise and factual."""


def _pretty(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _summarize_resume_content(content: str, source_name: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    top_lines = lines[:12]
    return "\n".join([f"Resume Summary: {source_name}", ""] + top_lines)


class LLMFileAssistant:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def answer(self, query: str) -> str:
        if self.client is None:
            return self._answer_without_llm(query)
        return self._answer_with_openai(query)

    def _answer_with_openai(self, query: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or "No response generated."

            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                tool = TOOLS[tool_name]
                result = tool(**arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": _pretty(result),
                    }
                )

    def _answer_without_llm(self, query: str) -> str:
        lower_query = query.lower()
        base_dir = Path("sample_data/resumes").resolve()

        if "read all resumes" in lower_query:
            files = list_files(str(base_dir))
            contents = []
            for entry in files:
                result = read_file(entry["path"])
                contents.append(
                    {
                        "file": entry["name"],
                        "success": result["success"],
                        "content_preview": result.get("content", "")[:220],
                    }
                )
            return _pretty(contents)

        if "find resumes mentioning" in lower_query:
            keyword = query.split("mentioning", maxsplit=1)[-1].strip(" .")
            keyword = re.sub(r"\bexperience\b", "", keyword, flags=re.IGNORECASE).strip() or keyword
            return self._search_resumes(base_dir, keyword)

        if "create a summary file for" in lower_query:
            match = re.search(r"for\s+([^\n]+)", query, flags=re.IGNORECASE)
            if not match:
                return "Could not determine which file to summarize."
            filename = match.group(1).strip().strip('"').strip("'")
            target = base_dir / filename
            if not target.exists():
                return f"File not found: {target}"
            return self._create_summary(target)

        if lower_query.startswith("list"):
            return _pretty(list_files(str(base_dir)))

        return (
            "OpenAI API key not configured. Supported local commands include:\n"
            '- "Read all resumes in the resumes folder"\n'
            '- "Find resumes mentioning Python experience"\n'
            '- "Create a summary file for resume_john_doe.txt"'
        )

    def _search_resumes(self, directory: Path, keyword: str) -> str:
        matches: list[dict[str, Any]] = []
        for entry in list_files(str(directory)):
            result = search_in_file(entry["path"], keyword)
            if result.get("match_count", 0) > 0:
                matches.append(
                    {
                        "file": entry["name"],
                        "match_count": result["match_count"],
                        "matches": result["matches"][:3],
                    }
                )
        return _pretty({"keyword": keyword, "results": matches})

    def _create_summary(self, filepath: Path) -> str:
        read_result = read_file(str(filepath))
        if not read_result.get("success"):
            return _pretty(read_result)

        summaries_dir = Path("generated_summaries").resolve()
        output_path = summaries_dir / f"{filepath.stem}_summary.txt"
        summary = _summarize_resume_content(read_result["content"], filepath.name)
        write_result = write_file(str(output_path), summary)
        return _pretty(
            {
                "summary_file": str(output_path),
                "write_result": write_result,
                "summary_preview": summary,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume file assistant with optional LLM tool calling.")
    parser.add_argument("query", nargs="?", help="Natural language query for the assistant.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    assistant = LLMFileAssistant(model=args.model)
    print(assistant.answer(args.query))


if __name__ == "__main__":
    main()
