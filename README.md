# claudeHelp
add env var ANTHROPIC_API_KEY

# Run as CLI
python workflow_llm.py "create a new site called portfolio" 

python workflow_llm.py "add a contact page"

# Run as API server
python workflow_llm.py api

curl -X POST "http://localhost:8000/api/hax-workflow" \
     -H "Content-Type: application/json" \
     -d '{"query": "create a new site called portfolio"}'
