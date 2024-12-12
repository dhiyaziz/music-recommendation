```markdown
# Music Recommendation System

This repository contains a music recommendation system based on Spotify's dataset. The project utilizes data analysis, clustering, and machine learning techniques to recommend songs based on user inputs.

## Features

- **Data Visualization**: Using `matplotlib`, `seaborn`, and `plotly`, the system provides insights into the Spotify dataset.
- **Clustering**: Songs and genres are grouped using K-Means clustering and visualized using t-SNE and PCA.
- **Feature Analysis**: Analysis of song features to understand their correlations with popularity.
- **Song Recommendation**: Recommends songs based on user input using feature-based similarity.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Spotify API credentials:
   - Create a `.env` file in the root directory with the following:
     ```env
     SPOTIFY_CLIENT_ID=your_client_id
     SPOTIFY_CLIENT_SECRET=your_client_secret
     ```

## Data

The project uses the following datasets:
- `data.csv`: Contains information about individual songs.
- `data_by_genres.csv`: Contains aggregated data by music genres.
- `data_by_year.csv`: Contains aggregated data by years.

## Usage

### Data Loading
The datasets are loaded into Pandas DataFrames:
```python
data = pd.read_csv('data.csv')
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')
```

### Feature Analysis
Correlations between features and popularity are visualized using `yellowbrick`:
```python
from yellowbrick.target import FeatureCorrelation

features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence', 'duration_ms', 'explicit', 'key', 'mode', 'year']
visualizer = FeatureCorrelation(labels=features)
visualizer.fit(X, y)
visualizer.show()
```

### Clustering
- **Genre Clustering**: Genres are clustered into 10 groups using K-Means.
- **t-SNE Visualization**:
  ```python
  from sklearn.manifold import TSNE
  tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2))])
  genre_embedding = tsne_pipeline.fit_transform(X)
  ```

- **PCA Visualization**:
  ```python
  from sklearn.decomposition import PCA
  pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
  song_embedding = pca_pipeline.fit_transform(X)
  ```

### Recommendation
Recommend songs similar to user input:
```python
recommend_songs(
    [{'name': 'Come As You Are', 'year': 1991},
     {'name': 'Smells Like Teen Spirit', 'year': 1991}], 
    data
)
```

## Results

- Clusters and visualizations provide insights into music trends by genre and year.
- Recommendations are personalized based on song attributes and user preferences.

## Requirements

- Python 3.7+
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `plotly`
  - `sklearn`
  - `scipy`
  - `yellowbrick`
  - `spotipy`
  - `python-dotenv`

## License

This project is licensed under the MIT License.

## Acknowledgments

- Kaggle: [Music Recommendation System Dataset](https://www.kaggle.com/datasets)
- Spotify API
```
