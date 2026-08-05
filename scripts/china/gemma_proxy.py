import itertools

import httpx
from fastapi import FastAPI, Request, Response


app = FastAPI()
targets = itertools.cycle(f"http://127.0.0.1:{port}" for port in range(40100, 40104))
client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512),
)


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(request: Request, path: str) -> Response:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k not in {"host", "content-length"}}
    query = f"?{request.url.query}" if request.url.query else ""
    for _ in range(2):
        try:
            response = await client.request(
                request.method, f"{next(targets)}/{path}{query}", content=body, headers=headers
            )
            if response.status_code < 500:
                return Response(
                    response.content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type"),
                )
        except httpx.HTTPError:
            pass
    return Response("Gemma replicas unavailable", status_code=503)
