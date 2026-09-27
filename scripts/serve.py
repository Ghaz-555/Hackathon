"""Serve the built React app and API together at http://127.0.0.1:8001."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import uvicorn
if __name__ == '__main__':
    if not (ROOT / 'frontend/dist/index.html').exists():
        raise SystemExit('Build the website first: cd frontend && npm install && npm run build')
    parser = argparse.ArgumentParser(description='Serve both Glasswork explorers and their API.')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run('server.app:app', host='127.0.0.1', port=args.port, workers=1)
