import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=".venv/bin/python", args=["ps_mcp_server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await s.call_tool("ps_login", {})
            print("LOGIN:", res.content[0].text.replace("\n"," ")[:200])
            res2 = await s.call_tool("ps_status", {})
            print("STATUS:", res2.content[0].text.replace("\n"," ")[:300])

asyncio.run(main())
