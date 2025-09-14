import sys
import os
from typing import List, Optional
import logging
from retriever.unified_retriever import query_code, index_code
import json
import requests
from dotenv import load_dotenv

# load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CodeQAInterface:
    """Interactive Q&A system for code quality queries using Anthropic Claude."""
    
    def __init__(self):
        """Initialize the Q&A interface with Anthropic Claude."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.context_chunks = 5  # Number of code chunks to retrieve
        self.api_base = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment. Please set it in .env file.")
        else:
            self._test_anthropic_connection()
    
    def _test_anthropic_connection(self):
        """Test if Anthropic API key is valid."""
        try:
            # test with a simple message using requests
            response = self._make_api_call("Hi", max_tokens=10)
            if response:
                logger.info(f"✅ Anthropic Claude ({self.model_name}) is ready!")
            else:
                logger.error("❌ Failed to connect to Anthropic API")
        except Exception as e:
            logger.error(f"❌ Error testing Anthropic connection: {e}")
        
    def _make_api_call(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Make a direct API call to Anthropic using requests."""
        if not self.api_key:
            return None
            
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": self.api_key
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        
        try:
            # make direct HTTP request without any proxy
            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=30,
                proxies={}  # Empty dict to explicitly avoid proxy
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
        
    def _query_claude(self, prompt: str) -> Optional[str]:
        """
        Query Claude with the given prompt.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Claude's response or None if error
        """
        if not self.api_key:
            return self._placeholder_llm_response(prompt)
            
        response = self._make_api_call(prompt)
        if response:
            return response
        else:
            return self._placeholder_llm_response(prompt)
    
    def _placeholder_llm_response(self, prompt: str) -> str:
        """
        Placeholder response when Claude is not available.
        
        Args:
            prompt: The prompt (for context)
            
        Returns:
            A placeholder response
        """
        return (
            "📝 Analysis Summary:\n"
            "Based on the retrieved code chunks, here are the key insights:\n"
            "• The code implements the requested functionality\n"
            "• Consider reviewing the implementation for best practices\n"
            "• Additional context may be needed for a more detailed analysis\n\n"
            "Note: This is a placeholder response. Set ANTHROPIC_API_KEY in .env for actual Claude analysis."
        )
    
    def _format_prompt(self, question: str, code_chunks: List[str]) -> str:
        """
        Format the prompt for Claude with question and relevant code.
        
        Args:
            question: The user's question
            code_chunks: Relevant code chunks from the retriever
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a senior software engineer analyzing code. Answer the following question based on the provided code chunks.

Question: {question}

Relevant Code Chunks:
"""
        for i, chunk in enumerate(code_chunks, 1):
            prompt += f"\n--- Code Chunk {i} ---\n{chunk}\n"
        
        prompt += "\nProvide a clear, concise answer focusing on the question asked. Include specific code references when relevant."
        
        return prompt
    
    def answer_question(self, question: str) -> str:
        """
        Process a question and return an answer using embeddings and Claude.
        
        Args:
            question: The user's question about the code
            
        Returns:
            The answer string
        """
        # retrieve relevant code chunks
        logger.info("🔍 Searching for relevant code...")
        code_chunks = query_code(question, k=self.context_chunks)
        
        if not code_chunks:
            return "❌ No relevant code chunks found. Please ensure code has been indexed."
        
        logger.info(f"✅ Found {len(code_chunks)} relevant code chunks")
        
        # format prompt and query LLM
        prompt = self._format_prompt(question, code_chunks)
        
        logger.info("🤖 Analyzing code with Claude and generating answer...")
        answer = self._query_claude(prompt)
        
        if answer:
            return answer
        else:
            return "❌ Failed to generate answer. Please check Claude API connection."
    
    def run_interactive_session(self):
        """Run the interactive Q&A session."""
        print("\n" + "="*60)
        print("🚀 Code Quality Intelligence Agent - Interactive Q&A")
        print("   Powered by Anthropic Claude")
        print("="*60)
        print("\nCommands:")
        print("  • Type your question about the code")
        print("  • Type 'quit' or 'exit' to end the session")
        print("  • Type 'help' for more information")
        print("\n" + "-"*60 + "\n")
        
        while True:
            try:
                # get user input
                question = input("❓ Your question: ").strip()
                
                # check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Code Quality Q&A. Goodbye!")
                    break
                
                # check for help command
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                # skip empty questions
                if not question:
                    continue
                
                # process the question
                print("\n" + "-"*60)
                answer = self.answer_question(question)
                print("\n💡 Answer:")
                print(answer)
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print("❌ An error occurred. Please try again.")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
📚 Help Information
==================

This interactive Q&A system allows you to ask questions about your codebase.
The system uses semantic search to find relevant code chunks and then uses
Claude to generate comprehensive answers.

Example Questions:
  • "How does the authentication system work?"
  • "What are the main security vulnerabilities in this code?"
  • "Explain the data flow in the payment module"
  • "What design patterns are used in this codebase?"
  • "How is error handling implemented?"

Tips:
  • Be specific in your questions for better results
  • The system searches through indexed code chunks
  • Ensure your code has been properly indexed before querying

Configuration:
  • Using Anthropic Claude API (direct HTTP)
  • Model: claude-3-haiku-20240307 (fast and efficient)
  • Retrieves top 5 most relevant code chunks by default
"""
        print(help_text)


def main():
    """Main entry point for the Q&A interface."""
    # check if code has been indexed
    try:
        test_chunks = query_code("test", k=1)
        if not test_chunks:
            print("⚠️  Warning: No code appears to be indexed yet.")
            print("   Run the main code quality agent first to index your codebase.")
            print()
    except Exception:
        print("⚠️  Warning: Code index not initialized.")
        print("   Run the main code quality agent first to index your codebase.")
        print()
    
    # create and run the Q&A interface
    qa_interface = CodeQAInterface()
    qa_interface.run_interactive_session()


if __name__ == "__main__":
    main()