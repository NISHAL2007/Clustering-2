# Clustering-2

A comprehensive clustering analysis project featuring an interactive Gradio web application for data visualization and K-Means clustering with multiple optimization techniques.

## 📋 Project Purpose

This project provides an end-to-end clustering pipeline designed to:
- Perform exploratory data analysis on datasets
- Apply K-Means clustering with intelligent optimization
- Visualize clustering results interactively
- Determine optimal cluster numbers using the Elbow Method and Silhouette Score
- Provide an intuitive web interface powered by Gradio

## 📊 Data

The project works with tabular datasets containing numerical features suitable for clustering analysis. Users can:
- Upload their own CSV files
- Analyze datasets with multiple numerical columns
- Automatically handle feature scaling and preprocessing

Supported data formats:
- CSV files with numerical features
- Data with any number of features (dimensionality reduction applied as needed)
- Missing values are handled through preprocessing

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NISHAL2007/Clustering-2.git
cd Clustering-2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Launch Gradio Interface (Recommended)

```bash
python app.py
```

This will start the Gradio web interface. Open your browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`).

#### Option 2: Run Clustering Pipeline Directly

```python
import pandas as pd
from clustering_pipeline import ClusteringPipeline

# Load your data
data = pd.read_csv('your_dataset.csv')

# Initialize and run pipeline
pipeline = ClusteringPipeline(data)
results = pipeline.fit_predict(n_clusters=3)

# Visualize results
pipeline.plot_clusters()
```

## 📦 Dependencies

Core libraries required:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
gradio>=4.0.0
scipy>=1.7.0
```

Additional dependencies:
- `plotly` - For interactive visualizations
- `joblib` - For model persistence
- `pillow` - For image processing in Gradio

### Installing Dependencies

Create a `requirements.txt` file with the above packages and run:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn gradio scipy plotly joblib pillow
```

## 🎯 Application Usage

### Using the Gradio Interface

1. **Upload Data**: Click the upload button and select your CSV file
2. **Configure Parameters**:
   - Choose number of clusters (or use auto-detection)
   - Select features to include in analysis
   - Adjust clustering parameters
3. **Run Analysis**: Click "Run Clustering" to execute the pipeline
4. **View Results**:
   - Cluster assignments for each data point
   - Visualization plots (scatter plots, elbow curves)
   - Cluster statistics and metrics
   - Download processed results

### Clustering Pipeline Features

#### 1. Elbow Method
Automatically determines optimal number of clusters by analyzing within-cluster sum of squares (WCSS).

#### 2. Silhouette Analysis
Evaluates cluster quality and separation using silhouette coefficients.

#### 3. Interactive Visualization
- 2D/3D scatter plots of clusters
- Color-coded cluster assignments
- Centroid markers
- Interactive zoom and pan

#### 4. Preprocessing Pipeline
- Automatic feature scaling (StandardScaler)
- Handling missing values
- Dimensionality reduction (PCA) for visualization
- Feature selection options

## 📖 Example Workflow

```python
# Complete example
import pandas as pd
import gradio as gr
from clustering_pipeline import ClusteringPipeline

# Load data
df = pd.read_csv('sample_data.csv')

# Initialize pipeline
cluster = ClusteringPipeline(data=df, n_clusters='auto')

# Fit and predict
labels = cluster.fit_predict()

# Get optimal k using elbow method
optimal_k = cluster.find_optimal_clusters(max_k=10)

# Visualize
cluster.plot_elbow_curve()
cluster.plot_silhouette_scores()
cluster.plot_clusters_2d()

# Export results
results_df = cluster.get_results()
results_df.to_csv('clustering_results.csv', index=False)
```

## 🛠️ Project Structure

```
Clustering-2/
│
├── app.py                  # Gradio application entry point
├── clustering_pipeline.py  # Core clustering logic
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── data/                  # Sample datasets
│   └── sample_data.csv
├── models/                # Saved models
└── outputs/               # Generated plots and results
```

## 🔧 Configuration

Customize clustering parameters in your code:

```python
pipeline = ClusteringPipeline(
    data=df,
    n_clusters=5,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
```

## 📊 Performance Metrics

The pipeline calculates:
- **Inertia**: Within-cluster sum of squares
- **Silhouette Score**: Measure of cluster separation (-1 to 1)
- **Davies-Bouldin Index**: Cluster similarity metric
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

NISHAL2007

## 🙏 Acknowledgments

- Built with scikit-learn for robust clustering algorithms
- Gradio for the intuitive web interface
- The open-source community for excellent libraries and tools

## 📞 Support

For questions or issues:
- Open an issue in the GitHub repository
- Contact: [Your contact information]

---

**Happy Clustering! 🎯📊**
