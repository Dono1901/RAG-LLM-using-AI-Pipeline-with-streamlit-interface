*RAG & LLM with Pathway to Process Financial Reports and Tables*

*Overview*

This documentation provides a technical guide for building a system that leverages Retrieval-Augmented Generation (RAG) and large language models (LLMs) to process financial reports and tables. The system integrates structured data (e.g., tables) and unstructured data (e.g., text, images) with Claude Sonet 3.5, using Pathway’s Google Drive Connector and Streamlit for the user interface.

*Key Components and Technologies*

*1\. Pathway Framework*

- *RAG Pipelines:* Pathway’s efficient pipeline framework is used to build scalable workflows for real-time and batch processing of multimodal data.
- *Google Drive Connector:* Handles real-time data ingestion from Google Drive for seamless integration of financial reports.
- *VectorStore/DocumentStore:* Enables instant access and retrieval of relevant data for analysis.

*2\. Claude Sonet 3.5*

- *LLM:* A powerful large language model used for natural language understanding and generation to analyze textual and tabular data.
- *Multimodal Processing:* Handles diverse data formats, enhancing the insights derived from financial reports.

*3\. Streamlit*

- *Frontend:* Provides an interactive user interface for visualizing financial data insights and interacting with the system.

*Folder Structure*

/pathway/financial-report-insights

├── app.py # Backend processing script with Pathway pipeline.

├── app.yaml # Pathway configuration for deployment.

├── Dockerfile # Dockerfile for containerization.

├── requirements.txt # Python dependencies.

├── credentials.json # Google Drive credentials.

├── streamlit_app.py # Streamlit frontend for user interaction.

└── claude_llm.py # Contains ClaudeSonetLLM class for interacting with Claude Sonet's API.

*Description of Files*

- *app.py:*
  - Implements the main Pathway pipeline.
  - Connects to Google Drive for ingesting financial reports.
  - Processes structured and unstructured data using RAG and Claude Sonet 3.5.
- *app.yaml:*
  - Contains deployment configurations for running the Pathway pipeline in production or test environments.
- *Dockerfile:*
  - Configures the application’s environment for containerized deployment.
  - Ensures compatibility and portability across systems.
- *requirements.txt:*
  - Specifies Python dependencies, including pathway, streamlit, and other necessary libraries.
- *credentials.json:*
  - Stores authentication credentials for accessing Google Drive.
- *streamlit_app.py:*
  - Builds an interactive frontend using Streamlit.
  - Displays financial insights and enables user input for specific queries.
- *claude_llm.py:*
  - Defines the ClaudeSonetLLM class, encapsulating the interaction logic with Claude Sonet’s API.
  - Supports multimodal data queries and responses.

*Implementation Steps*

*1\. Setting Up Pathway*

- Clone the repository:
- git clone <https://github.com/pathwaycom/llm-app.git>
- cd llm-app/examples/pipelines/gpt_4o_multimodal_rag

*2\. Install Pathway:*

- pip install pathway

*3\. Configuring Google Drive Connector*

- Place your credentials.json file in the /pathway/financial-report-insights folder.
- Update the app.py script to include the Google Drive connector:

from pathway.connectors.google_drive import GoogleDriveConnector

drive_connector = GoogleDriveConnector(credentials_path="credentials.json")

pipeline.add_source(drive_connector)

*4\. Building the Pipeline*

- Use the RAG pipeline logic from the [GitHub example](https://github.com/pathwaycom/pathway).
- Integrate structured and unstructured data processing using the ClaudeSonetLLM class in claude_llm.py.

*4\. Developing the Streamlit App*

- Create the streamlit_app.py script to visualize insights:

import streamlit as st

st.title("Financial Report Insights")

st.sidebar.file_uploader("Upload Financial Reports")

\# Add visualization and query components

*5\. Deploying the Application*

- Build the Docker image:

docker build -t financial-report-insights .

- Deploy using the Pathway CLI or Docker Compose.

*Conclusion*

By combining Pathway’s real-time processing capabilities with the Claude Sonet 3.5 LLM, this system enables efficient analysis of multimodal financial data. The use of Streamlit enhances accessibility and usability, providing a comprehensive tool for financial insights.
