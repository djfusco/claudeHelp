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

curl -X POST "http://localhost:8000/api/hax-workflow" \
     -H "Content-Type: application/json" \
     -d '{"query": "create a new site called portfolio"}'


# Sample output from curl
{"commands":["hax site start --name portfolio --y","cd portfolio","hax site node:add --title 'about' --y"],"descriptions":["Create new Hax site","Change to portfolio directory","Add new page with title 'about'"],"explanation":"Commands to execute:\n• Create new Hax site: hax site start --name portfolio --y\n• Change to portfolio directory: cd portfolio\n• Add new page with title 'about': hax site node:add --title 'about' --y\n\nSpecial notes:\n• Create new site named 'portfolio'\n• Add an 'about' page to the newly created site","confidence":0.8}
