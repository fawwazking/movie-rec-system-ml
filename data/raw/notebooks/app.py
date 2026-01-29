import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.sparse as sp


# PAGE CONFIGURATION

st.set_page_config(
    page_title="movie Recommendation system",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Letterboxd-style dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #14181c 0%, #1b2228 100%);
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #00e054 0%, #40bcf4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c2228 0%, #14181c 100%);
        border-right: 1px solid #2c3440;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #9ab !important;
    }
    
    /* Card styling for metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #242c34 0%, #1c2228 100%);
        border: 1px solid #3c4850;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00e054;
    }
    
    [data-testid="stMetric"] label {
        color: #9ab !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Text and paragraph colors */
    .stMarkdown, p, span {
        color: #9ab;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00e054 0%, #00c24a 100%);
        color: #14181c !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ff5f 0%, #00e054 100%);
        transform: translateY(-2px);
    }
    
    /* Selectbox and text input styling */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background-color: #2c3440 !important;
        border: none !important;
        border-radius: 25px !important;
        color: #9ab !important;
        transition: all 0.3s ease;
        padding-left: 20px !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover {
        background-color: #3c4850 !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        background-color: #3c4850 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #00e054 !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: #242c34;
        border-radius: 10px;
        padding: 10px 15px;
        border: 1px solid #3c4850;
    }
    
    .stRadio label {
        color: #9ab !important;
    }
    
    /* DataFrame / Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] > div {
        background: #1c2228;
        border-radius: 12px;
        border: 1px solid #3c4850;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #242c34 0%, #1c2228 100%) !important;
        border: 1px solid #3c4850 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: #1c2228;
        border: 1px solid #3c4850;
        border-top: none;
        border-radius: 0 0 10px 10px;
    }
    
    /* Divider styling */
    hr {
        border-color: #3c4850 !important;
        margin: 2rem 0;
    }
    
    /* Success/Warning/Error alerts */
    .stSuccess {
        background: linear-gradient(145deg, #1a3d23 0%, #14281a 100%);
        border: 1px solid #00e054;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(145deg, #3d3a1a 0%, #282614 100%);
        border: 1px solid #f0b000;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(145deg, #3d1a1a 0%, #281414 100%);
        border: 1px solid #ff6060;
        border-radius: 10px;
    }
    
    /* Info box in sidebar */
    .stAlert {
        background: #242c34;
        border: 1px solid #456;
        border-radius: 10px;
    }
    
    /* Caption styling */
    .stCaption {
        color: #678 !important;
    }
    
    /* Custom movie card styling */
    .movie-card {
        background: linear-gradient(145deg, #2c3440 0%, #242c34 100%);
        border: 1px solid #3c4850;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #00e054;
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .movie-meta {
        color: #9ab;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    
    .movie-rating {
        display: inline-block;
        background: linear-gradient(135deg, #00e054 0%, #00c24a 100%);
        color: #14181c;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .movie-tag {
        display: inline-block;
        background: #3c4850;
        color: #9ab;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Stats banner */
    .stats-banner {
        background: linear-gradient(145deg, #242c34 0%, #1c2228 100%);
        border: 1px solid #3c4850;
        border-radius: 16px;
        padding: 20px 30px;
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        color: #00e054;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #678;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Header section */
    .header-section {
        background: linear-gradient(135deg, #1c2228 0%, #14181c 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        border: 1px solid #3c4850;
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00e054 0%, #40bcf4 50%, #ff8000 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00e054 transparent transparent transparent !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #14181c;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3c4850;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00e054;
    }
    
    /* Tooltip enhancement */
    [data-testid="stTooltipIcon"] {
        color: #9ab !important;
    }
    
    /* Feature badge */
    .feature-badge {
        background: linear-gradient(135deg, #40bcf4 0%, #00a8e8 100%);
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load recommendation model and movie data"""
    try:
        # Use Path to get the correct directory
        current_dir = Path(__file__).parent
        processed_dir = current_dir / '../processed'
        
        movies = pd.read_csv(processed_dir / 'movie_list.csv')
        similarity = sp.load_npz(processed_dir / 'similarity_model.npz')
        return movies, similarity
    except FileNotFoundError as e:
        st.error("Error: Model files not found. Run 02_recommendation_system notebook first.")
        st.info(f"Looking for: ../processed/movie_list.csv and similarity_model.npz\n\nError: {str(e)}")
        return None, None

movies, similarity = load_data()

if movies is None or similarity is None:
    st.stop()


# HEADER & DESCRIPTION

# Header with premium styling
st.markdown("""
<div class="header-section" style="text-align: center;">
    <h1 style="margin: 0; font-size: 3rem;">Movie Finder</h1>
</div>
""", unsafe_allow_html=True)

# Stats banner
# Calculate unique genres and themes
unique_genres = len(set(', '.join(movies['genre'].dropna()).split(', ')))
unique_themes = len(set(', '.join(movies['themes'].dropna()).split(', ')))
min_year = int(movies['date'].min()) if movies['date'].notna().any() else 1920
max_year = int(movies['date'].max()) if movies['date'].notna().any() else 2024

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div style="text-align: center; padding: 15px;">
        <div style="color: #00e054; font-size: 2.2rem; font-weight: 700;">{len(movies):,}</div>
        <div style="color: #678; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Films</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 15px;">
        <div style="color: #40bcf4; font-size: 2.2rem; font-weight: 700;">{unique_genres}</div>
        <div style="color: #678; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Genres</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div style="text-align: center; padding: 15px;">
        <div style="color: #ff8000; font-size: 2.2rem; font-weight: 700;">{unique_themes}</div>
        <div style="color: #678; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Themes</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div style="text-align: center; padding: 15px;">
        <div style="color: #e040fb; font-size: 2.2rem; font-weight: 700;">{min_year}-{max_year}</div>
        <div style="color: #678; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Years</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# RECOMMENDATION ENGINE

def get_recommendations(movie_title, top_n=10):
    """
    Get movie recommendations based on similarity
    
    Args:
        movie_title: Movie title to search for
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommendations or tuple (None, error_message)
    """
    try:
        # Search for exact match (case-insensitive)
        matching_idx = movies[movies['name'].str.lower() == movie_title.lower()]
        
        if len(matching_idx) == 0:
            # Try partial match
            partial_idx = movies[movies['name'].str.contains(movie_title, case=False, na=False)]
            if len(partial_idx) == 0:
                return None, f"Movie '{movie_title}' not found in database"
            idx = partial_idx.index[0]
            found_exact = False
        else:
            idx = matching_idx.index[0]
            found_exact = True
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity[idx].toarray().ravel()))
        sim_scores = sorted(sim_scores, reverse=True, key=lambda x: x[1])
        
        # Get top N (exclude the selected movie)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        top_scores = [i[1] for i in sim_scores[1:top_n+1]]
        
        # Build DataFrame
        recommendations = movies.iloc[top_indices][['name', 'genre', 'themes', 'rating', 'cast', 'date', 'movie_era']].copy()
        recommendations['Match Score'] = [f"{score:.4f}" for score in top_scores]
        recommendations['No'] = range(1, len(recommendations) + 1)
        recommendations = recommendations[['No', 'name', 'genre', 'themes', 'rating', 'cast', 'date', 'movie_era', 'Match Score']]
        
        return recommendations, found_exact
    
    except Exception as e:
        return None, f"Error: {str(e)}"


# SIDEBAR - SETTINGS

with st.sidebar:
    # Sidebar image
    image_path = Path(__file__).parent / "garry.jpg"
    st.image(str(image_path), use_container_width=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <h2 style="margin: 10px 0 0 0; color: #ffffff;">Movie Finder</h2>
        <p style="color: #678; font-size: 0.85rem;">Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<p style="color: #9ab; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem;">SETTINGS</p>', unsafe_allow_html=True)
    
    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
    st.markdown("---")
    
    st.markdown('<p style="color: #9ab; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem; margin-bottom: 5px;">PRO TIPS</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0;">
        <p style="color: #9ab; font-size: 0.85rem; margin: 0 0 6px 0;">Type movie title accurately for best results</p>
        <p style="color: #9ab; font-size: 0.85rem; margin: 0 0 6px 0;">Partial matches also work</p>
        <p style="color: #9ab; font-size: 0.85rem; margin: 0 0 6px 0;">Higher score = more similar</p>
        <p style="color: #9ab; font-size: 0.85rem; margin: 0;">Results in milliseconds</p>
    </div>
    """, unsafe_allow_html=True)


# SEARCH INPUT

st.markdown("""
<div style="margin-bottom: 20px;">
    <h2 style="color: #ffffff; margin-bottom: 5px;">Search for a Movie</h2>
    <p style="color: #678; font-size: 0.9rem;">Find your favorite film and discover similar titles</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    search_type = st.radio(
        "Search method:",
        ["Browse", "Search"],
        horizontal=True
    )

if search_type == "Browse":
    selected_movie = st.selectbox(
        'Select movie:',
        options=sorted(movies['name'].unique()),
        key="dropdown_select"
    )
    search_input = selected_movie
else:
    search_input = st.text_input(
        "Movie title:",
        placeholder="e.g., Inception, Fight Club, Parasite",
        key="text_search"
    )


# ACTION BUTTON & RESULTS

st.markdown("""
<style>
    div.stButton > button[kind="primary"] {
        background: #3a9c35 !important;
        color: #14181c !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 1rem 2rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        border-radius: 10px !important;
        min-height: 60px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button('FIND RECOMMENDATIONS', use_container_width=True, type="primary"):
    
    if not search_input:
        st.warning("⚠️ Please enter a movie title!")
    else:
        with st.spinner("🔄 Searching for similar movies..."):
            recommendations, found_exact = get_recommendations(search_input, top_n=num_recommendations)
        
        if recommendations is None:
            st.error(f"❌ {found_exact}")
            
            # Alternative suggestions
            st.markdown("""
            <div style="margin-top: 20px;">
                <h3 style="color: #ff8000;">🤔 Did you mean:</h3>
            </div>
            """, unsafe_allow_html=True)
            
            similar_titles = movies[movies['name'].str.contains(search_input, case=False, na=False)]['name'].head(5)
            if len(similar_titles) > 0:
                for title in similar_titles:
                    st.markdown(f"""
                    <div style="background: #242c34; padding: 10px 15px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #40bcf4;">
                        <span style="color: #ffffff;">🎬 {title}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No similar titles found")
        
        else:
            # Display selected movie info
            selected_film = movies[movies['name'].str.lower() == search_input.lower()]
            if len(selected_film) == 0:
                selected_film = movies[movies['name'].str.contains(search_input, case=False, na=False)].iloc[0:1]
            
            if len(selected_film) > 0:
                film = selected_film.iloc[0]
                
                # Selected movie header with custom card
                if found_exact:
                    st.markdown("""
                    <div style="background: #1a2e1a; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;">
                        <span style="color: #00e054; font-weight: 500;">Perfect Match Found</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #2e2a1a; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;">
                        <span style="color: #f0b000; font-weight: 500;">Using Partial Match</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Movie info card
                st.markdown(f"""
                <div class="movie-card" style="background: linear-gradient(145deg, #2c3440 0%, #1c2228 100%);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 250px;">
                            <h2 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.8rem;">{film['name']}</h2>
                            <p style="color: #678; margin: 0 0 15px 0;">
                                <span style="color: #40bcf4;">📅 {int(film['date']) if pd.notna(film['date']) else 'N/A'}</span>
                                &nbsp;&nbsp;•&nbsp;&nbsp;
                                <span style="color: #ff8000;">🎭 {film['genre'][:40] + ('...' if len(str(film['genre'])) > 40 else '')}</span>
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: linear-gradient(135deg, #00e054 0%, #00c24a 100%); color: #14181c; padding: 8px 20px; border-radius: 25px; font-weight: 700; font-size: 1.2rem; display: inline-block;">
                                ⭐ {film['rating'] if pd.notna(film['rating']) else 'N/A'}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Display recommendations header
            st.markdown(f"""
            <div style="margin: 20px 0;">
                <h2 style="color: #ffffff; margin-bottom: 5px;"> Top {len(recommendations)} Recommended Movies</h2>
                <p style="color: #678; font-size: 0.9rem;">Based on genre and theme similarity</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Rename columns for display
            display_df = recommendations.rename(columns={
                'name': 'Film Title',
                'genre': 'Genre',
                'themes': 'Themes',
                'rating': 'Rating',
                'cast': 'Cast',
                'date': 'Year',
                'movie_era': 'Era',
                'Match Score': 'Similarity'
            })
            
            # Display table with formatting
            st.dataframe(
                display_df.set_index('No'),
                use_container_width=True,
                height=450
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Recommendation statistics with custom styling
            st.markdown("""
            <div style="margin: 20px 0 10px 0;">
                <h3 style="color: #ffffff;">📊 Recommendation Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_rating = pd.to_numeric(recommendations['rating'], errors='coerce').mean()
                st.metric("Avg Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
            with col2:
                avg_score = pd.to_numeric(recommendations['Match Score'].str.replace('nan', '0'), errors='coerce').mean()
                st.metric("Avg Match Score", f"{avg_score:.4f}")
            with col3:
                total_genres = len(set(', '.join(recommendations['genre'].dropna()).split(', ')))
                st.metric("Unique Genres", total_genres)
            with col4:
                st.metric("Total Results", len(recommendations))


# FAQ & ADDITIONAL INFO

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("FAQ - Frequently Asked Questions"):
    st.markdown("""
**Q: How does this system work?**

Uses Content-Based Filtering with TF-IDF Vectorization and Cosine Similarity. Each film is analyzed by genre and themes to find similar movies.

**Q: Is this machine learning?**

Yes! It uses traditional unsupervised learning techniques for text analysis and similarity computation.

**Q: How accurate is it?**

Accuracy depends on data quality. More detailed metadata = better results.

**Q: What data is used?**

Dataset contains movie information including genre, themes, rating, cast, and release year.
    """)

with st.expander("Technical Details"):
    st.markdown(f"""
**Dataset**
- Total Films: {len(movies):,}
- Features: Genre and Themes
- Data: Rating, Cast, Year, Era

**Algorithm**
- TF-IDF Vectorization for text feature extraction
- Cosine Similarity for measuring similarity
- Sparse Matrix for memory efficiency

**Performance**
- Query Time: <100ms per search
- Similarity Matrix: Pre-computed
    """)


# FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <p style='color: #456; font-size: 1rem;'>
        <span style="color: #9ab;">Movie Recommendation System</span>
    </p>
</div>
""", unsafe_allow_html=True)