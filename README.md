# Decentralized Learning System with Adaptive Federated Learning

A robust implementation of a decentralized learning system that combines machine learning with expert rules and SHAP explanations. The system implements adaptive federated learning with multiple clients.

## Features

- **Decentralized Learning**: Multiple clients train on local data
- **Adaptive Federated Learning**: Dynamic learning rate adjustment
- **Expert System Integration**: Rule-based decision support
- **SHAP Explanations**: Model interpretability
- **Real-time Monitoring**: Comprehensive logging and metrics

## Project Structure

```
project/
├── clients/                 # Client implementations
│   ├── client1.py
│   ├── client2.py
│   └── client3.py
├── common/                  # Shared utilities
│   ├── expert_system.py
│   ├── model.py
│   └── shap_explanation.py
├── models/                  # Saved models
├── data/                    # Dataset
├── server.py               # Central server
└── requirements.txt        # Dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the System

1. Start the server:
```bash
python server.py
```

2. Start clients (in separate terminals):
```bash
python clients/client1.py client1
python clients/client2.py client2
python clients/client3.py client3
```

## Results

The system generates various results including:
- Model performance metrics
- SHAP visualizations
- Expert rule explanations
- Training logs
- Federated learning statistics

## Requirements

- Python 3.8+
- LightGBM
- SHAP
- Flask
- NumPy
- Pandas
- Scikit-learn

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 