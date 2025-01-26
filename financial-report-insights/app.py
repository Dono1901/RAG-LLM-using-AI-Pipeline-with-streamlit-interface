import logging
import os
from dotenv import load_dotenv
from pathway.xpacks.llm.question_answering import SummaryQuestionAnswerer
from pathway.xpacks.llm.servers import QASummaryRestServer
from pydantic import BaseModel, ConfigDict, InstanceOf

# Import the custom ClaudeSonetLLM class
from claude_llm import ClaudeSonetLLM

# Setup Pathway
import pathway as pw
pw.set_license_key("demo-license-key-with-telemetry")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()

class App(BaseModel):
    question_answerer: InstanceOf[SummaryQuestionAnswerer]
    host: str = "0.0.0.0"
    port: int = 8000

    with_cache: bool = True
    terminate_on_error: bool = False

    def run(self) -> None:
        server = QASummaryRestServer(self.host, self.port, self.question_answerer)
        server.run(
            with_cache=self.with_cache,
            terminate_on_error=self.terminate_on_error,
        )

    model_config = ConfigDict(extra="forbid")


if __name__ == "__main__":
    with open("app.yaml") as f:
        config = pw.load_yaml(f)
    app = App(**config)

    # Initialize the ClaudeSonet LLM (with your API Key here)
    claude_api_key = "sk-a**"  # Replace with your actual Claude Sonet API key
    claude_llm = ClaudeSonetLLM(api_key=claude_api_key)

    # Initialize the Question Answerer with Claude Sonet
    question_answerer = SummaryQuestionAnswerer(llm=claude_llm)

    app.run()
