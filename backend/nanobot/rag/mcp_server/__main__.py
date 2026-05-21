"""Entry point for RAG MCP Server.

Run with: python -m nanobot.rag.mcp_server
       or: python -m nanobot.rag.mcp_server --config /path/to/settings.yaml
"""
import argparse
import sys

from nanobot.rag.mcp_server.server import run_stdio_server


def main() -> int:
    parser = argparse.ArgumentParser(description="Modular RAG MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml file",
    )
    args, unknown = parser.parse_known_args()

    if args.config:
        # Set environment variable for settings path
        import os
        os.environ["RAG_SETTINGS_PATH"] = args.config

    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
