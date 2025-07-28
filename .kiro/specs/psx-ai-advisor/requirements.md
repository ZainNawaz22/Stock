# Requirements Document

## Introduction

The PSX AI Advisor is a personal, automated data analysis tool designed to empower investment decisions on the Pakistan Stock Exchange (PSX). The system will systematically gather and analyze stock data by downloading the daily "Closing Rate Summary" PDF from dps.psx.com.pk/downloads, leveraging machine learning to provide simple, actionable insights. This tool serves as a data-driven co-pilot for navigating the PSX, focusing on daily stock data extraction from PDF files, technical indicator calculation, and predictive modeling to generate investment suggestions.

## Requirements

### Requirement 1: Data Acquisition System

**User Story:** As a data-driven investor, I want the system to automatically download and extract daily stock data from the PSX Closing Rate Summary PDF, so that I have comprehensive up-to-date market information for analysis.

#### Acceptance Criteria

1. WHEN the user executes the data acquisition script THEN the system SHALL download the daily "Closing Rate Summary" PDF from dps.psx.com.pk/downloads
2. WHEN the PDF is downloaded THEN the system SHALL extract OHLCV (Open, High, Low, Close, Volume) data for all stocks from the PDF
3. WHEN a network error occurs during download THEN the system SHALL retry the request at least once before failing
4. WHEN the data extraction process runs THEN the system SHALL complete data acquisition for all stocks within reasonable time limits

### Requirement 2: Technical Indicator Calculation

**User Story:** As a data-driven investor, I want the system to calculate key technical indicators from the scraped data, so that I have enriched data for better analysis.

#### Acceptance Criteria

1. WHEN raw OHLCV data is available THEN the system SHALL calculate 50-day Simple Moving Average (SMA_50)
2. WHEN raw OHLCV data is available THEN the system SHALL calculate 200-day Simple Moving Average (SMA_200)
3. WHEN raw OHLCV data is available THEN the system SHALL calculate 14-day Relative Strength Index (RSI_14)
4. WHEN raw OHLCV data is available THEN the system SHALL calculate Moving Average Convergence Divergence (MACD)
5. WHEN technical indicators are calculated THEN the system SHALL append them as new columns to the daily data

### Requirement 3: Data Storage and Management

**User Story:** As a data-driven investor, I want the system to store historical and new data persistently, so that I can build up a database for trend analysis and model training.

#### Acceptance Criteria

1. WHEN enriched data is processed THEN the system SHALL save it to a CSV file named after the stock symbol (e.g., ENGRO.csv)
2. WHEN a CSV file already exists for a stock symbol THEN the system SHALL append new data without overwriting historical data
3. WHEN data is stored THEN the CSV file SHALL contain all OHLCV columns plus calculated technical indicators
4. WHEN the system runs for 7 consecutive days THEN the system SHALL maintain data integrity across all stored files

### Requirement 4: Machine Learning Prediction System

**User Story:** As a data-driven investor, I want the system to use machine learning to predict next-day price movements, so that I can make informed investment decisions.

#### Acceptance Criteria

1. WHEN sufficient historical data exists in a stock's CSV file THEN the system SHALL train a machine learning model on that data
2. WHEN the model is trained THEN the system SHALL predict whether the next day's closing price will be higher or lower than the current day's close
3. WHEN a prediction is generated THEN the system SHALL output a human-readable suggestion (e.g., "Prediction for ENGRO: UP")
4. WHEN the model makes predictions THEN the system SHALL use Random Forest or equivalent algorithm for classification

### Requirement 5: Command-Line Interface

**User Story:** As a data-driven investor, I want to trigger the entire analysis process with a single command, so that I can easily run daily analysis without complex setup.

#### Acceptance Criteria

1. WHEN the user runs the main Python script THEN the system SHALL execute PDF download, data extraction, processing, and prediction in sequence
2. WHEN the script is running THEN the system SHALL print status updates to the terminal (e.g., "Downloading PDF...", "Extracting data for ENGRO...", "Calculating indicators...")
3. WHEN the process completes THEN the system SHALL display final predictions for all processed stocks
4. WHEN an error occurs THEN the system SHALL provide clear error messages to help with troubleshooting

### Requirement 6: System Reliability and Performance

**User Story:** As a data-driven investor, I want the system to be reliable and performant, so that I can depend on it for daily market analysis.

#### Acceptance Criteria

1. WHEN the system processes all KSE100 stocks THEN the entire daily process SHALL complete within reasonable time limits
2. WHEN network errors occur THEN the system SHALL handle them gracefully with retry logic
3. WHEN the system runs THEN the code SHALL be well-commented and organized into logical functions
4. WHEN the system operates for 7 consecutive days THEN it SHALL successfully generate predictions without errors