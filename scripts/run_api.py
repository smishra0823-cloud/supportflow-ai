import os

import uvicorn


def main() -> None:
    host = os.getenv("SUPPORTFLOW_HOST", "127.0.0.1")
    port = int(os.getenv("SUPPORTFLOW_API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
