"""
JSON-RPC 2.0 server.

Dispatches to RPCHandlers. Supports both HTTP POST and WebSocket.
Port 8545 by default (eth-compatible).
"""

from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import web

from .handlers import RPCHandlers

log = logging.getLogger("chain.rpc.server")


class RPCServer:
    def __init__(self, sequencer, shard_protocol=None, benchmark=None, thought_store=None, host: str = "0.0.0.0", port: int = 8545):
        self._host     = host
        self._port     = port
        self._handlers = RPCHandlers(sequencer, shard_protocol, benchmark, thought_store)

    async def run(self) -> None:
        app = web.Application(middlewares=[self._cors_middleware])
        app.router.add_post("/", self._handle_http)
        app.router.add_route("OPTIONS", "/", self._handle_options)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/health", self._handle_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        log.info("rpc_server_started host=%s port=%d", self._host, self._port)

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler) -> web.Response:
        resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    async def _handle_options(self, req: web.Request) -> web.Response:
        return web.Response(headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    async def _handle_health(self, req: web.Request) -> web.Response:
        result = await self._handlers.inft_getChainInfo([])
        return web.json_response({"status": "ok", **result})

    async def _handle_http(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        result = await self._dispatch(body)
        return web.json_response(result)

    async def _handle_ws(self, req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    body   = json.loads(msg.data)
                    result = await self._dispatch(body)
                    await ws.send_json(result)
                except Exception as exc:
                    await ws.send_json({"error": str(exc)})
        return ws

    async def _dispatch(self, body: dict | list) -> dict | list:
        if isinstance(body, list):
            return [await self._dispatch_one(req) for req in body]
        return await self._dispatch_one(body)

    async def _dispatch_one(self, req: dict) -> dict:
        rpc_id  = req.get("id", None)
        method  = req.get("method", "")
        params  = req.get("params", [])

        # Route method to handler
        handler_name = method.replace(".", "_").replace("-", "_")
        handler = getattr(self._handlers, handler_name, None)

        if handler is None:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:
            log.debug("rpc_error method=%s err=%s", method, exc)
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": str(exc)}}
