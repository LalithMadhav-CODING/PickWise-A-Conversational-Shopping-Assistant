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
import json
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from langchain_core.tools import StructuredTool

from . import config, shop

_server_process = None


def local_tools() -> dict:
    return {f.__name__: StructuredTool.from_function(f) for f in shop.TOOL_FUNCTIONS}


def server_is_up() -> bool:
    """True when OUR server answers (its /health route), not just any program that holds the port."""
    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/health"
    try:
        with build_opener(ProxyHandler({})).open(Request(url), timeout=2) as response:
            return json.load(response).get("service") == "pickwise-catalogue"
    except (OSError, ValueError, URLError):
        return False


def _startup_error(message: str) -> RuntimeError:
    """The error shows the end of the server's log, so a failed start explains itself in the notebook (on Colab
    the log file is out of sight)."""
    path = config.ROOT / "mcp_server.log"
    tail = path.read_text(errors="replace")[-4000:] if path.exists() else "(no server log)"
    if "address already in use" in tail.lower():
        message += (f"\nPort {config.MCP_PORT} is taken, probably by a server left over from an earlier session: "
                    "stop it (`!pkill -f Pickwise.mcp_server`) or set PICKWISE_MCP_PORT, then run this cell again.")
    return RuntimeError(f"{message}\nServer log ({path}), last lines:\n{tail}")


def ensure_server(timeout: float = 120) -> str:
    """Start the MCP server in the background unless it already runs. The server opens its port within seconds and
    loads the index and models after that (mcp_server.py), so this only waits for the port."""
    global _server_process
    if server_is_up():
        return "already running"
    if _server_process is None or _server_process.poll() is not None:
        with open(config.ROOT / "mcp_server.log", "w") as log:
            _server_process = subprocess.Popen([sys.executable, "-u", "-m", "Pickwise.mcp_server"],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT)
        atexit.register(_server_process.terminate)        # a server we started stops when we stop
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_process.poll() is not None:
            raise _startup_error("The MCP server exited during start-up.")
        if server_is_up():
            return "started"
        time.sleep(0.5)
    _server_process.terminate()
    try:
        _server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _server_process.kill()
        _server_process.wait(timeout=5)
    raise _startup_error(f"The MCP server did not become healthy at {config.MCP_URL} within {timeout:.0f}s.")


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
