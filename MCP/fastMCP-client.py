import asyncio
import json
from fastmcp import Client

MCP_URL = "http://localhost:8000/mcp"


async def main():
    client = Client(MCP_URL)

    async with client:
        print(f"\n🔗 Connected to MCP server: {MCP_URL}")

        # List tools once
        tools = await client.list_tools()
        print("\n🧰 Tools:")
        for t in tools:
            print("  -", t.name)

        # Interactive query loop
        while True:
            query = input("\nEnter a query (or 'exit' to quit): ").strip()
            if query.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            if not query:
                continue

            try:
                top_n_str = input("How many results? [default 5]: ").strip()
                top_n = int(top_n_str) if top_n_str else 5
            except ValueError:
                top_n = 5

            print(f"\n🚀 Calling semantic_search_mdsoar with query: {query!r} ...")

            resp = await client.call_tool(
                "semantic_search_mdsoar",
                {"query": query, "top_n": top_n},
            )

            if not resp.content:
                print("No content returned from tool.")
                continue

            block = resp.content[0]
            if hasattr(block, "text"):
                try:
                    results = json.loads(block.text)
                except json.JSONDecodeError:
                    print("Got text content but it wasn't valid JSON:")
                    print(block.text)
                    continue
            else:
                print("Unexpected content type:", type(block), block)
                continue

            if not results:
                print("No results.")
                continue

            print("\n📄 Search results:")
            for r in results:
                print(f"  • {r['filename']} (score: {r['score']:.4f}, method: {r['method']})")


if __name__ == "__main__":
    asyncio.run(main())
