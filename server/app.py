"""
server/app.py - OpenEnv multi-mode deployment server entry point.
Required by openenv validate for multi-mode deployment spec.

This file MUST have:
  - main() function
  - if __name__ == '__main__': block calling main()
"""

import os
import sys

# Add project root to Python path so src/ and app.py are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """
    Server entry point called by openenv validate and [project.scripts].
    Starts the ClinicalTrialRecruiter HTTP server on port 7860.
    """
    from app import run_server
    port = int(os.getenv("PORT", "7860"))
    run_server(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
