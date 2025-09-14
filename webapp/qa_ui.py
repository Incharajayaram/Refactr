from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/qa/{job_id}", response_class=HTMLResponse)
async def qa_interface(job_id: str):
    """Serve the Q&A interface for a specific job."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Q&A - Job {job_id}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }}
            .header {{
                background: white;
                padding: 1rem 2rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .container {{
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: 1200px;
                margin: 0 auto;
                width: 100%;
                padding: 2rem;
                gap: 2rem;
            }}
            .chat-container {{
                flex: 1;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            .messages {{
                flex: 1;
                overflow-y: auto;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}
            .message {{
                padding: 1rem;
                border-radius: 8px;
                max-width: 80%;
            }}
            .user-message {{
                background-color: #0366d6;
                color: white;
                align-self: flex-end;
            }}
            .bot-message {{
                background-color: #f8f9fa;
                border: 1px solid #e1e4e8;
                align-self: flex-start;
            }}
            .input-container {{
                padding: 1.5rem;
                border-top: 1px solid #e1e4e8;
                display: flex;
                gap: 1rem;
            }}
            #questionInput {{
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 1rem;
            }}
            #askButton {{
                background-color: #0366d6;
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 4px;
                font-size: 1rem;
                cursor: pointer;
            }}
            #askButton:hover {{
                background-color: #0256c7;
            }}
            #askButton:disabled {{
                background-color: #ccc;
                cursor: not-allowed;
            }}
            .status {{
                padding: 1rem;
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                margin-bottom: 1rem;
            }}
            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #0366d6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .code-block {{
                background-color: #f6f8fa;
                border: 1px solid #e1e4e8;
                border-radius: 4px;
                padding: 1rem;
                margin: 0.5rem 0;
                overflow-x: auto;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 0.9em;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Code Q&A Assistant</h1>
            <div>
                <span>Job ID: <code>{job_id}</code></span>
                <a href="/" style="margin-left: 2rem;">← Back to Home</a>
            </div>
        </div>
        
        <div class="container">
            <div id="status" class="status">
                ⏳ Checking index status...
            </div>
            
            <div class="chat-container">
                <div id="messages" class="messages">
                    <div class="message bot-message">
                        👋 Welcome! I'm ready to answer questions about your codebase. 
                        Ask me anything about the code structure, functionality, or implementation details.
                    </div>
                </div>
                
                <div class="input-container">
                    <input type="text" id="questionInput" placeholder="Ask a question about the code..." disabled>
                    <button id="askButton" disabled>Ask</button>
                </div>
            </div>
        </div>
        
        <script>
            const jobId = '{job_id}';
            const messagesDiv = document.getElementById('messages');
            const statusDiv = document.getElementById('status');
            const questionInput = document.getElementById('questionInput');
            const askButton = document.getElementById('askButton');
            
            let indexReady = false;
            
            // Check if index exists or create it
            async function checkOrCreateIndex() {{
                try {{
                    // First check if index exists
                    const statusResponse = await fetch(`/api/qa/status/${{jobId}}`);
                    
                    if (statusResponse.ok) {{
                        const status = await statusResponse.json();
                        if (status.status === 'ready') {{
                            indexReady = true;
                            statusDiv.innerHTML = '✅ Index ready! Ask your questions below.';
                            questionInput.disabled = false;
                            askButton.disabled = false;
                            return;
                        }}
                    }}
                    
                    // If not ready, create index
                    statusDiv.innerHTML = '🔄 Creating code index for Q&A... This may take a minute.';
                    
                    // Get job info to get repository URL
                    const jobResponse = await fetch(`/api/jobs/${{jobId}}`);
                    if (!jobResponse.ok) {{
                        throw new Error('Failed to get job info');
                    }}
                    const jobInfo = await jobResponse.json();
                    
                    // Create index
                    const indexResponse = await fetch('/api/qa/index', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            repository_url: jobInfo.repo_url || window.location.origin,
                            job_id: jobId
                        }})
                    }});
                    
                    if (!indexResponse.ok) {{
                        throw new Error('Failed to create index');
                    }}
                    
                    // Poll for index status
                    const pollIndex = async () => {{
                        const response = await fetch(`/api/qa/status/${{jobId}}`);
                        const status = await response.json();
                        
                        if (status.status === 'ready') {{
                            indexReady = true;
                            statusDiv.innerHTML = `✅ Index ready! Indexed ${{status.chunks_indexed}} code chunks.`;
                            questionInput.disabled = false;
                            askButton.disabled = false;
                        }} else if (status.status === 'failed') {{
                            statusDiv.innerHTML = `❌ Index creation failed: ${{status.error}}`;
                        }} else {{
                            setTimeout(pollIndex, 2000);
                        }}
                    }};
                    
                    pollIndex();
                    
                }} catch (error) {{
                    statusDiv.innerHTML = `❌ Error: ${{error.message}}`;
                }}
            }}
            
            // Format code blocks in messages
            function formatMessage(text) {{
                // Replace code blocks with styled divs
                return text.replace(/```([\\s\\S]*?)```/g, '<div class="code-block">$1</div>')
                          .replace(/`([^`]+)`/g, '<code>$1</code>')
                          .replace(/\\n/g, '<br>');
            }}
            
            // Send question
            async function askQuestion() {{
                const question = questionInput.value.trim();
                if (!question || !indexReady) return;
                
                // Disable input
                questionInput.disabled = true;
                askButton.disabled = true;
                
                // Add user message
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.textContent = question;
                messagesDiv.appendChild(userMessage);
                
                // Add loading message
                const loadingMessage = document.createElement('div');
                loadingMessage.className = 'message bot-message';
                loadingMessage.innerHTML = '<div class="loading"></div> Analyzing code and generating answer...';
                messagesDiv.appendChild(loadingMessage);
                
                // Scroll to bottom
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
                try {{
                    const response = await fetch('/api/qa/query', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            question: question,
                            job_id: jobId,
                            k: 5
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Failed to get answer');
                    }}
                    
                    const data = await response.json();
                    
                    // Replace loading message with answer
                    loadingMessage.innerHTML = formatMessage(data.answer);
                    
                    // Add metadata
                    const metadata = document.createElement('div');
                    metadata.style.fontSize = '0.8em';
                    metadata.style.color = '#666';
                    metadata.style.marginTop = '0.5rem';
                    metadata.textContent = `Found ${{data.relevant_chunks}} relevant code chunks • Response time: ${{data.response_time.toFixed(2)}}s`;
                    loadingMessage.appendChild(metadata);
                    
                }} catch (error) {{
                    loadingMessage.innerHTML = `❌ Error: ${{error.message}}`;
                }} finally {{
                    // Re-enable input
                    questionInput.value = '';
                    questionInput.disabled = false;
                    askButton.disabled = false;
                    questionInput.focus();
                    
                    // Scroll to bottom
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }}
            }}
            
            // Event listeners
            askButton.addEventListener('click', askQuestion);
            questionInput.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    askQuestion();
                }}
            }});
            
            // Initialize
            checkOrCreateIndex();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)