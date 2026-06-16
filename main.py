import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "promptmigrator.api:app",
        host=os.getenv("PM_HOST", "127.0.0.1"),
        port=int(os.getenv("PM_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
