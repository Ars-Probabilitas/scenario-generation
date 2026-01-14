
# Create TinyTimeMixer (TTM) architecture diagram using Mermaid

diagram_code = """
flowchart TB
    subgraph pretrain["<b>Pre-training Phase (Univariate)</b>"]
        A1[Input: Time Series Data]
        A2[Normalization Layer]
        A3[Patching Module]
        A4["TTM Backbone<br/>Level 1"]
        A5["TTM Backbone<br/>Level 2"]
        A6["TTM Backbone<br/>Level 3"]
        A7[TTM Decoder]
        A8[Forecast Linear Head]
        A9[Output: Predictions]
        
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 --> A5
        A5 --> A6
        A6 --> A7
        A7 --> A8
        A8 --> A9
    end
    
    subgraph adaptive["<b>Adaptive Patching</b>"]
        P1["Level 1:<br/>[c, n, hf] → [c, 4n, hf/4]"]
        P2["Level 2:<br/>[c, n, hf] → [c, 16n, hf/16]"]
        P3["Level 3:<br/>[c, n, hf] → [c, 64n, hf/64]"]
        
        P1 --> P2
        P2 --> P3
    end
    
    subgraph finetune["<b>Fine-tuning Phase (Multivariate)</b>"]
        B1[Input: Multiple Channels]
        B2[Channel Mixing Module]
        B3[Exogenous Mixing Module<br/>optional]
        B4["Frozen TTM Backbone<br/>Level 1 🔒"]
        B5["Frozen TTM Backbone<br/>Level 2 🔒"]
        B6["Frozen TTM Backbone<br/>Level 3 🔒"]
        B7[Updated TTM Decoder]
        B8[Forecast Linear Head]
        B9[Output: Multi-channel Predictions]
        
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
        B6 --> B7
        B7 --> B8
        B8 --> B9
    end
    
    A3 -.-> P1
    P3 -.-> B3
    
    style A4 fill:#1FB8CD,stroke:#333,stroke-width:2px,color:#fff
    style A5 fill:#1FB8CD,stroke:#333,stroke-width:2px,color:#fff
    style A6 fill:#1FB8CD,stroke:#333,stroke-width:2px,color:#fff
    style A7 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    style A8 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    
    style B2 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    style B3 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    style B4 fill:#9FA8B0,stroke:#333,stroke-width:2px,color:#333
    style B5 fill:#9FA8B0,stroke:#333,stroke-width:2px,color:#333
    style B6 fill:#9FA8B0,stroke:#333,stroke-width:2px,color:#333
    style B7 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    style B8 fill:#2E8B57,stroke:#333,stroke-width:2px,color:#fff
    
    style P1 fill:#FFEB8A,stroke:#333,stroke-width:2px,color:#333
    style P2 fill:#FFEB8A,stroke:#333,stroke-width:2px,color:#333
    style P3 fill:#FFEB8A,stroke:#333,stroke-width:2px,color:#333
    
    style pretrain fill:#B3E5EC,stroke:#333,stroke-width:3px
    style finetune fill:#FFCDD2,stroke:#333,stroke-width:3px
    style adaptive fill:#FFEB8A,stroke:#333,stroke-width:3px
"""

# Create the diagram using the helper function
create_mermaid_diagram(diagram_code, 'ttm_architecture.png', 'ttm_architecture.svg', width=1600, height=1000)
