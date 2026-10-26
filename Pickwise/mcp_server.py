"""Pickwise catalogue MCP server. Adapted from Module 4 · Agentic_app (mcp_server/knowledge_server.py)
and MCP_Server_and_Tools_Lab.

Exposes the catalogue to ANY MCP client (this project's agents, Claude Desktop, an IDE, MCP Inspector):

    Tools     : search_products(query, category, k), get_product(item_id), list_categories()
    Resources : catalogue://stats

add_to_cart is deliberately NOT here: it is a high-risk action that only the cart node may run, after
the shopper approves (Phase 5).

Run it (from the project root):
    python -m Pickwise.mcp_server                     # streamable HTTP on 127.0.0.1:8766/mcp (used by the agents)
    python -m Pickwise.mcp_server --transport stdio   # for Claude Desktop / MCP Inspector
"""
import argparse
import json
import sys

from mcp.server.fastmcp import FastMCP

from . import config, shop

mcp = FastMCP("pickwise-catalogue", host=config.MCP_HOST, port=config.MCP_PORT)
for function in shop.TOOL_FUNCTIONS:
    mcp.tool()(function)


@mcp.resource("catalogue://stats")
def catalogue_stats() -> str:
    """How many products and categories the catalogue holds."""
    catalogue = shop.get_catalogue()
    return json.dumps({"products": len(catalogue), "categories": catalogue["category"].nunique()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    args = parser.parse_args()
    if args.transport == "http":
        shop.warm_up()                  # load the index and models once, before the first request
        print(f"Pickwise MCP server listening on {config.MCP_URL}", file=sys.stderr, flush=True)
    mcp.run(transport="streamable-http" if args.transport == "http" else "stdio")
