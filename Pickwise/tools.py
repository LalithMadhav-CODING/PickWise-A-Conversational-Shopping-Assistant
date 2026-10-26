"""The agents' tools, from either transport. Adapted from Module 4 · Agentic_app (tools.py, mcp_client.py).

    via="local" : the catalogue functions as in-process LangChain tools
    via="mcp"   : the same functions discovered from the MCP server (tools/list), called over HTTP

| Tool            | Type                   | Risk | Policy                                   |
|-----------------|------------------------|------|------------------------------------------|
| search_products | Retrieval (RAG)        | Low  | runs automatically                       |
| get_product     | Retrieval (lookup)     | Low  | runs automatically                       |
| list_categories | Retrieval (lookup)     | Low  | runs automatically                       |
| add_to_cart     | Action (Phase 5)       | HIGH | local only, after the shopper approves   |
"""
import atexit
import socket
import subprocess
import sys
import time

from langchain_core.tools import StructuredTool

from . import config, shop

_server_process = None


def local_tools() -> dict:
    return {f.__name__: StructuredTool.from_function(f) for f in shop.TOOL_FUNCTIONS}


def server_is_up() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((config.MCP_HOST, config.MCP_PORT)) == 0


def ensure_server(timeout: float = 300) -> str:
    """Start the MCP server in the background unless something already listens on its port."""
    global _server_process
    if server_is_up():
        return "already running"
    if _server_process is None or _server_process.poll() is not None:
        log = open(config.ROOT / "mcp_server.log", "w")
        _server_process = subprocess.Popen([sys.executable, "-m", "Pickwise.mcp_server"],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
        atexit.register(_server_process.terminate)        # a server we started stops when we stop
    deadline = time.time() + timeout                       # the first start loads the index and models
    while time.time() < deadline:
        if server_is_up():
            return "started"
        if _server_process.poll() is not None:
            raise RuntimeError("The MCP server exited during start-up. See mcp_server.log.")
        time.sleep(0.5)
    raise RuntimeError(f"The MCP server did not open {config.MCP_URL} within {timeout:.0f}s. See mcp_server.log.")


async def mcp_tools() -> dict:
    """tools/list over MCP -> {name: LangChain tool}. Nothing about these tools is hard-coded here."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    ensure_server()
    client = MultiServerMCPClient({"catalogue": {"url": config.MCP_URL, "transport": "streamable_http"}})
    return {t.name: t for t in await client.get_tools()}


async def get_tools(via: str = "mcp") -> dict:
    if via == "local":
        return local_tools()
    if via == "mcp":
        return await mcp_tools()
    raise ValueError("via must be 'mcp' or 'local'")


def tool_text(result) -> str:
    """Local tools return a string; MCP tools return content blocks. Normalise both to text."""
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):                         # (content, artifact) from some adapter versions
        result = result[0]
    return "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
