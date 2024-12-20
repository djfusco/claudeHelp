# dual_mode_hax_workflow.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import anthropic
import os
import subprocess
import sys
import json
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
import uvicorn
import typer

# Load environment variables
load_dotenv()

# Initialize FastAPI and Typer apps
app = FastAPI()
cli = typer.Typer()

# Rich console for better output formatting
console = Console()

@dataclass
class WorkflowResult:
    step: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None

class WorkflowRequest(BaseModel):
    query: str = Field(..., description="Natural language query describing what to create")

class WorkflowResponse(BaseModel):
    commands: List[str]
    descriptions: List[str]
    explanation: str
    confidence: float

class HaxSiteWorkflow:
    def __init__(self):
        """Initialize the Hax Site Workflow Agent with Anthropic client"""
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def create_system_prompt(self):
        """Create a detailed system prompt for the Hax CLI workflow"""
        return """You are an AI assistant that interprets natural language queries into specific Hax CLI commands.

        Available Workflows:
        1. Create a new website:
           - Command: hax site start --name [siteName] --y
           - Creates new directory with site
           - Opens localhost:3000

        2. Add new page:
           - First ensures we're in the correct site directory
           - Command: hax site node:add --title [pageTitle] --y
           - Creates new page in current site

        Respond in JSON format:
        {
            "workflow_type": "new_site" or "add_page",
            "site_name": "extracted_site_name",
            "page_title": "extracted_page_title",
            "confidence": 0.0 to 1.0,
            "special_instructions": ["any special notes"]
        }

        Examples:
        - "create a new site called portfolio" → {"workflow_type": "new_site", "site_name": "portfolio"}
        - "add an about page to my portfolio site" → {"workflow_type": "add_page", "site_name": "portfolio", "page_title": "about"}
        - "add a contact page" → {"workflow_type": "add_page", "page_title": "contact"}
        """

    def check_current_directory(self, site_name: str) -> bool:
        """Check if we're already in the correct directory"""
        try:
            current_dir = os.path.basename(os.getcwd())
            return current_dir == site_name
        except Exception as e:
            console.print(f"[red]Error checking directory: {str(e)}")
            return False

    def parse_query_with_llm(self, query: str) -> Dict:
        """Use Anthropic's Claude to parse and understand the query"""
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system=self.create_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"Parse this request and extract the workflow details: {query}"
                    }
                ]
            )
            
            content = message.content[0].text
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                error_msg = f"Error: LLM response was not valid JSON. Response was: {content}"
                if console.is_terminal:
                    console.print(f"[red]{error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing query with LLM: {str(e)}"
            if console.is_terminal:
                console.print(f"[red]{error_msg}")
            raise ValueError(error_msg)

    def validate_names(self, site_name: Optional[str], page_title: Optional[str]) -> None:
        """Validate site name and page title"""
        if site_name and not re.match(r'^[a-zA-Z0-9_]+$', site_name):
            raise ValueError("Invalid site name. Use only letters, numbers, and underscores.")
        
        if page_title and not re.match(r'^[a-zA-Z0-9_\s]+$', page_title):
            raise ValueError("Invalid page title. Use only letters, numbers, spaces, and underscores.")

    def execute_workflow(self, workflow_type: str, site_name: Optional[str], page_title: Optional[str]) -> List[WorkflowResult]:
        """Execute the specified workflow with provided parameters"""
        workflow_steps = []

        if workflow_type == "new_site":
            workflow_steps = [
                {
                    "description": "Create new Hax site",
                    "command": f"hax site start --name {site_name} --y"
                },
                {
                    "description": f"Change to {site_name} directory",
                    "command": f"cd {site_name}"
                }
            ]
            if page_title:
                workflow_steps.append({
                    "description": f"Add new page with title '{page_title}'",
                    "command": f"hax site node:add --title '{page_title}' --y"
                })
        elif workflow_type == "add_page":
            if site_name and not self.check_current_directory(site_name):
                if os.path.isdir(site_name):
                    workflow_steps.append({
                        "description": f"Change to {site_name} directory",
                        "command": f"cd {site_name}"
                    })
                else:
                    raise ValueError(f"Directory '{site_name}' not found. Make sure you're in the parent directory of your site.")

            workflow_steps.append({
                "description": f"Add new page with title '{page_title}'",
                "command": f"hax site node:add --title '{page_title}' --y"
            })

        workflow_results = []
        for step in workflow_steps:
            try:
                result = subprocess.run(
                    step['command'],
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                workflow_results.append(WorkflowResult(
                    step=step['description'],
                    status="success",
                    output=result.stdout
                ))
            except subprocess.CalledProcessError as e:
                workflow_results.append(WorkflowResult(
                    step=step['description'],
                    status="error",
                    error=str(e),
                    output=e.stderr
                ))
                break

        return workflow_results

def print_results(results: List[WorkflowResult], explanation: str, confidence: float):
    """Print workflow results in a nicely formatted way"""
    table = Table(title="Workflow Execution Results")
    table.add_column("Step", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Output/Error", style="white")

    for result in results:
        status_color = "green" if result.status == "success" else "red"
        output_text = result.output if result.status == "success" else result.error
        table.add_row(
            result.step,
            f"[{status_color}]{result.status}[/{status_color}]",
            str(output_text)
        )

    console.print(Panel(explanation, title="Workflow Plan", border_style="blue"))
    console.print(f"\nConfidence Score: [yellow]{confidence:.2f}[/yellow]")
    console.print(table)

# CLI Command
@cli.command()
def run(query: str = typer.Argument(..., help="Natural language query describing what to create")):
    """Execute Hax workflow from command line"""
    workflow = HaxSiteWorkflow()
    
    try:
        console.print("[yellow]Analyzing your request...[/yellow]")
        parsed = workflow.parse_query_with_llm(query)
        
        if parsed["confidence"] < 0.7:
            console.print("[red]I'm not confident I understand what you want to do. Could you rephrase your request?")
            raise typer.Exit(1)
        
        workflow.validate_names(
            parsed.get("site_name"),
            parsed.get("page_title")
        )
        
        console.print("[yellow]Executing workflow...[/yellow]")
        results = workflow.execute_workflow(
            workflow_type=parsed["workflow_type"],
            site_name=parsed.get("site_name"),
            page_title=parsed.get("page_title")
        )
        
        explanation = "Workflow execution plan:\n" + "\n".join(
            f"• {result.step}" for result in results
        )
        
        if "special_instructions" in parsed:
            explanation += "\n\nSpecial notes:\n" + "\n".join(
                f"• {note}" for note in parsed["special_instructions"]
            )
        
        print_results(results, explanation, parsed["confidence"])
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        raise typer.Exit(1)

# API Routes
@app.post("/api/hax-workflow", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowRequest):
    """Process natural language query and return required commands without executing them"""
    workflow = HaxSiteWorkflow()
    
    try:
        parsed = workflow.parse_query_with_llm(request.query)
        
        if parsed["confidence"] < 0.7:
            raise HTTPException(
                status_code=400,
                detail="Query unclear. Please rephrase your request."
            )
        
        workflow.validate_names(
            parsed.get("site_name"),
            parsed.get("page_title")
        )
        
        # Generate workflow steps without executing
        workflow_steps = []
        workflow_type = parsed["workflow_type"]
        site_name = parsed.get("site_name")
        page_title = parsed.get("page_title")

        if workflow_type == "new_site":
            workflow_steps = [
                {
                    "description": "Create new Hax site",
                    "command": f"hax site start --name {site_name} --y"
                },
                {
                    "description": f"Change to {site_name} directory",
                    "command": f"cd {site_name}"
                }
            ]
            if page_title:
                workflow_steps.append({
                    "description": f"Add new page with title '{page_title}'",
                    "command": f"hax site node:add --title '{page_title}' --y"
                })
        elif workflow_type == "add_page":
            if site_name:
                workflow_steps.append({
                    "description": f"Change to {site_name} directory",
                    "command": f"cd {site_name}"
                })
            workflow_steps.append({
                "description": f"Add new page with title '{page_title}'",
                "command": f"hax site node:add --title '{page_title}' --y"
            })
        
        explanation = "Commands to execute:\n" + "\n".join(
            f"• {step['description']}: {step['command']}" 
            for step in workflow_steps
        )
        
        if "special_instructions" in parsed:
            explanation += "\n\nSpecial notes:\n" + "\n".join(
                f"• {note}" for note in parsed["special_instructions"]
            )
        
        return WorkflowResponse(
            commands=[step["command"] for step in workflow_steps],
            descriptions=[step["description"] for step in workflow_steps],
            explanation=explanation,
            confidence=parsed["confidence"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/examples")
async def get_examples():
    """Return example queries that the API can handle"""
    return {
        "examples": [
            "create a new site called portfolio",
            "add an about page to my portfolio site",
            "add a contact page",
            "create a blog website",
            "add a new page called team to my company site"
        ]
    }

if __name__ == "__main__":
    # Check if running as API or CLI
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as CLI
        cli()