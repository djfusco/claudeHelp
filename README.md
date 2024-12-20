# set up local python environment
python -m venv venv

(on Windows)
venv\Scripts\activate

OR 

(on Mac/Linux)
source venv/bin/activate


pip install -r requirements.txt



# Environment Variable needed
add env var ANTHROPIC_API_KEY to local .env



# Run as CLI
python workflow_llm.py "create a new site called portfolio" 

python workflow_llm.py "add a contact page"

# Run as API server
python workflow_llm.py api

curl -X POST "http://localhost:8000/api/workflow_llm" \
     -H "Content-Type: application/json" \
     -d '{"query": "create a new site called portfolio"}'
