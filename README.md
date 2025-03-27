# Agentic Ticket Analysis System

An AI-powered solution for generating qualitative insights from IT service management ticket data.

## Overview

This system analyzes service desk ticket data to provide actionable insights, automation opportunities, and intelligent responses to user queries. The application is built using a modular, agent-based architecture that leverages large language models (LLMs) for advanced analysis.

## Features

- **Data Upload**: Support for CSV and Excel (.xls, .xlsx) ticket data files.
- **Intelligent Data Processing**: Automatic preprocessing, cleaning, and feature engineering.
- **Comprehensive Insights**: AI-generated analysis of ticket patterns, trends, and key statistics.
- **Interactive Visualizations**: Charts and graphs to visualize ticket data patterns.
- **Predefined Questions**: AI-generated questions and answers about the ticket data, including automation potential.
- **Automation Opportunities**: Detailed analysis of top automation possibilities with implementation plans.
- **Interactive Chat**: Query your ticket data using natural language.

## Architecture

The system consists of the following modules:

1. **Data Extraction**: Extracts and loads ticket data from uploaded files.
2. **Data Processing**: Cleans, transforms, and enriches the ticket data.
3. **Insight Generator**: Generates insights and patterns from the processed data.
4. **Chart Generator**: Creates visualizations from the processed data.
5. **Automation Analyzer**: Identifies and details automation opportunities.
6. **Query Processor**: Processes natural language queries about the ticket data.

## Installation

### Prerequisites

- Python 3.8 or higher
- Streamlit
- GROQ API access

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ticket-analysis-system.git
   cd ticket-analysis-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory with your GROQ API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama3-70b-8192
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Upload Ticket Data**: Use the file uploader in the left sidebar to upload your ticket data file (CSV or Excel).
2. **Explore Insights**: Navigate to the "Data Insights" tab to see generated insights and visualizations.
3. **View Predefined Questions**: Check the "Predefined Questions" tab for AI-generated questions and answers about your data.
4. **Explore Automation Opportunities**: Visit the "Automation Opportunities" tab to see detailed automation suggestions.
5. **Chat with Your Data**: Use the "Chat with Your Data" tab to ask natural language questions about your ticket data.

## Data Format

The system is designed to work with typical IT service desk ticket data. While it can handle various column names, the following fields are commonly used:

- Ticket Number
- Priority
- Opened date
- Assignment Group
- Subcategory
- Short description
- State
- Opened by
- Assigned to
- Closed date
- Work notes
- Resolution notes

## Scaling for Large Datasets

The system automatically handles large datasets by processing them in chunks (default: 1000 rows). This ensures that even large ticket exports can be analyzed without memory issues.

## Customization

You can customize the system by:

1. Modifying the `.env` file to change configuration settings
2. Editing the chart generation in `chart_generator.py` to add new visualization types
3. Extending the data processing module in `data_processing.py` to add new features or transformations

## Troubleshooting

- **File Upload Issues**: Ensure your CSV or Excel file has headers and follows standard formats.
- **API Connection Issues**: Check that your GROQ API key is correctly set in the `.env` file.
- **Memory Errors**: For very large datasets, try increasing the chunk size in the `.env` file.
