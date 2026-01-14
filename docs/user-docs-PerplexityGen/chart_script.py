
# Create the Doob Decomposition with TinyTimeMixer workflow flowchart

diagram_code = """
flowchart TD
    Start["Time Series Data (X_t)"]
    Doob["Doob Decomposition"]
    Martingale["Martingale Component (M_t)<br/><small>Unpredictable, random fluctuations</small>"]
    Drift["Drift Component (A_t)<br/><small>Predictable, systematic trend</small>"]
    TTM["TTM Model<br/><small>TinyTimeMixer Pre-trained Model</small>"]
    Forecast["Forecast Output<br/><small>• Point predictions<br/>• Prediction intervals<br/>• Component forecasts</small>"]
    Evaluation["Evaluation & Analysis"]
    
    Start --> Doob
    Doob --> Martingale
    Doob --> Drift
    Martingale --> TTM
    Drift --> TTM
    TTM --> Forecast
    Forecast --> Evaluation
"""

# Create the mermaid diagram using the helper function
create_mermaid_diagram(diagram_code, 'doob_decomposition_flowchart.png', 'doob_decomposition_flowchart.svg')
