import streamlit as st
import kagglehub as kh
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot
import geopandas as gpd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# University Headers
st.markdown("""
<div style='text-align: center'>
    <h4>BUCHAREST UNIVERSITY OF ECONOMIC STUDIES</h4>
    <h5>CYBERNETICS, STATISTICS AND ECONOMIC INFORMATICS FACULTY</h5>
    <p><b>- Software Packages Project -</b></p>
    <h2>Classic Alt. Rock tracks analysis</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Coordinator:**  
Belciu Anda

**Carried out by:**  
Ciucu-Barcan Theodor-George  
Craciun Stefan

**Group:** 1104
""")

st.divider()

st.write("""
For this project we decided to use a dataset focused on alternative rock in order to showcase the different functionalities and practices using python and various libraries such as streamlit, geopandas, numpy, etc.

In order to import the dataset we opted to use the kagglehub library instead of manually downloading it from their website. Our datasets makes use of various characteristics collected from Spotify in order to give a deeper understanding analysis of the tracks and how they are different from each other without having to listen to every single one of them.

We also decided to host the application using the integrated free hosting feature by linking a github account on the official Streamlit app.
""")

st.title("Classic Alt Rock Dataset Analysis")

@st.cache_data
def load_data():
    path = kh.dataset_download("thebumpkin/800-classic-alt-rock-tracks-with-spotify-data")
    csv_file_path = os.path.join(path, "ClassicAltRock.csv")
    df = pd.read_csv(csv_file_path)
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.subheader("1. Raw Dataset Preview")
st.write("""
The first challenge for this project was initializing Streamlit and importing the dataset. The raw dataset has a total of 781 rows and 18 columns.
We used the kagglehub package, Streamlit and the python environment to initialize our app.

**Calculation methods/algorithms:** 
- `@st.cache_data` to optimize loading times
- `kagglehub.dataset_download()` to fetch the data
- `pd.read_csv()` to read the CSV file into a structured dataframe

Importing the dataset enables us to start our economic analysis on the Alt Rock dataset.
""")
st.dataframe(df.head(800))

st.subheader("Dataset Dimensions")
st.write(f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

#X 2. Extracted Artists
st.write("""
Extracting all unique artists from the dataset and displaying them cleanly as strings, sorted either alphabetically (A to Z) or dynamically by their average platform popularity.
We used streamlit methods (`st.radio`, `st.columns`, `st.divider`) and pandas for grouping.

**Calculation methods/algorithms:** 
- `.unique().tolist()` to isolate unique artists
- `df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)` to sort by popularity

Like most streaming platforms, Spotify pays artists by the number of streams in a month; the higher the average popularity is, the higher their revenue is. However, this revenue goes to the right holders (usually record labels and distributors) and the actual money the artists take home depends entirely on the specific contract they signed. But supposing that every artist in this data set would have the same exact contract this would be their hierarchy:
""")

st.write(f"There are a total of {len(artists_list)} unique artists present in the initial database:")

num_cols = 5
cols = st.columns(num_cols)

for i, artist in enumerate(artists_list):
    cols[i % num_cols].write(f"- {artist}")

st.divider()

#X 3. Filtered Dataset
st.subheader("3. Filtered Dataset")
st.write("""
The initial dataset offers us a great total of 95 artists; we decided to cut it down to a total of 40 artists to combat overplotting and have a better visual readability.
We used streamlit methods to display buttons, pandas to group and matplotlib for the bar charts.

**Calculation methods/algorithms:** 
- The highlight is the pandas `isin.selected_bands` filtering algorithm to estimate the revenue.
""")

st.write("A newly created dataset strictly isolating the targeted bands:")

selected_bands = [
    "3 Doors Down", "Alice In Chains", "Blur", "Counting Crows", "Dead Kennedys", 
    "Deftones", "Depeche Mode", "Disturbed", "Elvis Costello", "Everclear", 
    "Foo Fighters", "Green Day", "Incubus", "Joy Division", "King Crimson", "Korn", 
    "Linkin Park", "Muse", "my bloody valentine", "My Chemical Romance", "New Order", 
    "Nine Inch Nails", "Nirvana", "Oasis", "Papa Roach", "Pearl Jam", "Pet Shop Boys", 
    "Red Hot Chili Peppers", "Rob Zombie", "Sex Pistols", "Soundgarden", 
    "System Of A Down", "Talking Heads", "The Cars", 
    "The Clash", "The Cure", "The Smashing Pumpkins", "The Smiths", "TOOL", "Weezer"
]

filtered_df = df[df['Artist'].isin(selected_bands)].reset_index(drop=True)

st.dataframe(filtered_df)
st.write(f"The beautifully filtered dataset now has exactly {filtered_df.shape[0]} rows.")

st.write(f"**The {len(selected_bands)} Officially Filtered Artists:**")

sort_option_filtered = st.radio("Sort Filtered Artists By:", ["Alphabetical (A-Z)", "Average Popularity (High to Low)"], horizontal=True, key="sort_filtered")

if sort_option_filtered == "Average Popularity (High to Low)":
    artist_pop_f = filtered_df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)
    display_bands = artist_pop_f.index.tolist()
else:
    display_bands = sorted(selected_bands, key=str.lower)

num_cols_f = 5
cols_f = st.columns(num_cols_f)

for i, artist in enumerate(display_bands):
    cols_f[i % num_cols_f].write(f"- {artist}")

st.write("---")
st.write("**March 2026 Estimated Artist Revenue Comparison**")

if st.button("Show Revenue Chart"):
    with st.spinner("Generating revenue chart..."):
        fig_rev, ax_rev = matplotlib.pyplot.subplots(figsize=(8, 5))
        bands = ['Red Hot Chili Peppers', 'Sex Pistols']
        
        rhcp_revenue = 45844488 * 0.005
        pistols_revenue = 1258867 * 0.005
        revenues = [rhcp_revenue, pistols_revenue]
        
        bar_colors = ['red', 'black']
        
        ax_rev.bar(bands, revenues, color=bar_colors, edgecolor='black')
        
        ax_rev.set_ylabel("Revenue ($)", fontweight='bold')
        ax_rev.set_title("March 2026 Estimated Spotify Revenue", fontsize=14, fontweight='bold')
        ax_rev.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(revenues):
            ax_rev.text(i, v + 5000, f"${v:,.2f}", ha='center', fontweight='bold', fontsize=11)
            
        st.pyplot(fig_rev)

st.write("---")
st.write("**Average Popularity Ranking of the 40 Selected Bands**")

if st.button("Show Popularity Chart for 40 Bands"):
    with st.spinner("Generating popularity comparison chart..."):
        band_popularity = filtered_df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)
        
        fig_pop, ax_pop = matplotlib.pyplot.subplots(figsize=(14, 6))
        
        import matplotlib.cm as cm
        bar_colors = cm.viridis(np.linspace(0, 1, len(band_popularity)))
        
        ax_pop.bar(band_popularity.index, band_popularity.values, color=bar_colors, edgecolor='black')
        
        ax_pop.set_xticks(range(len(band_popularity)))
        ax_pop.set_xticklabels(band_popularity.index, rotation=90, fontsize=9)
        ax_pop.set_ylabel("Average Spotify Popularity", fontweight='bold')
        ax_pop.set_title("Average Spotify Popularity of the 40 Filtered Artists", fontsize=16, fontweight='bold')
        ax_pop.grid(axis='y', linestyle='--', alpha=0.6)
        
        st.pyplot(fig_pop)

st.write("""
The chart showcases the popularity hierarchy of the 40 selected bands and gives us a deeper understanding of how these artists perform in terms of listener engagement and streaming visibility on the platform. At the top of the ranking, bands like **Red Hot Chili Peppers** and **Linkin Park** stand out with the highest average popularity scores, close to 80. The middle section of the chart includes artists like **Weezer**, **Counting Crows**, and **Paramore** with a slightly lower popularity. Lastly, toward the lower end we have bands such as **The Cure**, **Joy Division** and **Sex Pistols**; these bands have a comparatively lower popularity score.

The graph also reveals high economic inequality on Spotify and streaming platforms in general. The average payout on Spotify is roughly $0.003 to $0.005 per stream; this means the estimated revenue generated by popular bands is significantly greater than niche bands that struggle to stay relevant in the eyes of the label. For example, Red Hot Chili Peppers has a revenue as high as $230,000 (estimated for March 2026) while the revenue generated by Sex Pistols is only as high as $6,250, making it almost 37 times as high.
""")

st.divider()

#X 4. Statistical Processing & Aggregation
st.subheader("4. Statistical Processing & Aggregation")
st.write("""
Calculating aggregated metrics (sum, mean, count) for audio features like "Danceability" and "Tempo" per artist, and testing if these features mathematically correlate with a song's popularity.
We used pandas to calculate the sum and mean for danceability, mean for Tempo and count for tracks, matplot for bar chart and correlation matrix. We also used `filtered_df[...].dropna()` to deal with any possible missing data.

Using Pandas `.groupby()` to calculate the **sum** and **mean** of `Danceability` for each artist's tracks:
""")


df_agg = filtered_df.groupby(['Artist']).agg({'Danceability': [sum, "mean"], 
                                              'Tempo': "mean",               
                                              'Track': 'count'})             

st.dataframe(df_agg)

if st.button("Show Aggregated Data Graph"):
    with st.spinner("Rendering Matplotlib chart..."):
        artist_names = df_agg.index
        avg_tempo = df_agg[('Tempo', 'mean')]
        
        fig3, ax3 = matplotlib.pyplot.subplots(figsize=(14, 6))
        
        import matplotlib.colors as mcolors
        
        all_colors = list(mcolors.cnames.keys())
        
        ax3.bar(artist_names, avg_tempo, color=all_colors[:len(artist_names)], edgecolor="black")
        
        ax3.set_xticks(range(len(artist_names)))
        ax3.set_xticklabels(artist_names, rotation=90, fontsize=9)
        ax3.set_xlabel("Artist", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Average Tempo (BPM)", fontsize=12, fontweight='bold')
        ax3.set_title("Average Track Tempo per Artist", fontsize=16)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig3)

st.write("""
This chart illustrates how much tempo (measured in BPM) can vary from a band to another band even if they are from the same music genre. We can point out higher-tempo artists such as **Green Day** and **Foo Fighters** which are usually aligned with punk influences. For mid-range bands such as **Red Hot Chili Peppers** and **Weezer**, their tempo reflects versatility. Lastly, the lower end is filled with artists such as **The Cure** and **Joy Division**; they tend to have a more melancholic musical style compared to the rest of the presented bands.
""")

st.write("---")
st.write("**Correlation Analysis: Danceability, Tempo, and Popularity**")

if st.button("Show Correlation Matrix"):
    with st.spinner("Calculating Pearson correlations..."):
        corr_data = filtered_df[['Danceability', 'Tempo', 'Popularity']].dropna()
        
        corr_matrix = corr_data.corr()
        
        st.write("Raw Mathematical Matrix (Color Coded):")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.3f}"))
        
        fig_corr, ax_corr = matplotlib.pyplot.subplots(figsize=(6, 5))
        
        cax = ax_corr.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig_corr.colorbar(cax, shrink=0.8)
        
        labels = ['Danceability', 'Tempo', 'Popularity']
        ax_corr.set_xticks(range(len(labels)))
        ax_corr.set_yticks(range(len(labels)))
        ax_corr.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax_corr.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax_corr.xaxis.set_ticks_position('bottom')
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax_corr.text(j, i, f"{val:.2f}",
                             ha="center", va="center", color=text_color, fontweight='bold', fontsize=12)
                
        ax_corr.set_title("Pearson Feature Correlation Heatmap", pad=20, fontsize=14, fontweight='bold')
        
        st.pyplot(fig_corr)
        
        st.write("""
The heatmap displays the Pearson correlation coefficient between three key audio features presented in the data set: **danceability**, **tempo**, and **popularity**:

- **Danceability vs. Tempo**: There is a weak negative correlation between how fast a song is (Tempo) and how danceable it is. As Tempo increases, Danceability tends to linearly decrease.
- **Danceability vs. Popularity**: There is no linear correlation. Danceability possesses zero predictive power regarding popularity and both variables are statistically linearly independent.
- **Tempo vs. Popularity**: There is an extremely weak, negligible negative linear correlation; the effect size is utterly negligible.

**In conclusion**, none of the features strongly influence each other. This means that danceability and tempo are not key to make a rock song successful, but other factors like marketing and artist reputation might play a bigger role.
""")
        st.info(" **Statistical Insight:** A correlation value of exactly `1.00` is perfect (e.g., Tempo vs Tempo). If Popularity strongly scales with Danceability, the number will be firmly positive. If it is close to `0.00`, there is zero linear mathematical relationship between them in this dataset.")

st.divider()

# 5. Merge / Join Datasets
st.subheader("5. Processing Datasets with Merge / Join")
st.write("We create a secondary standalone dataset containing the 'Country of Origin' for each artist, and use Pandas `pd.merge()` to mathematically join them:")

country_mapping = pd.DataFrame({
    'Artist': selected_bands,
    'Country': [
        "USA", "USA", "UK", "USA", "USA", 
        "USA", "UK", "USA", "UK", "USA", 
        "USA", "USA", "USA", "UK", "UK", "USA", 
        "USA", "UK", "Ireland", "USA", "UK", 
        "USA", "USA", "UK", "USA", "USA", "UK", 
        "USA", "USA", "UK", "USA", 
        "USA", "USA", "USA", 
        "UK", "UK", "USA", "UK", "USA", "USA"
    ]
})

st.dataframe(country_mapping)

merged_df = pd.merge(filtered_df, country_mapping, on='Artist', how='left')

st.dataframe(merged_df)

st.write("""
**Mapping the 40 filtered artists to their country of origin and visualizing how tracks are distributed globally.**
We create a secondary standalone second dataset containing the 'Country of Origin' for each artist, and then used Pandas `pd.merge()` to mathematically join them. We also used GeoPandas for the function `gpd.read_file()` to fetch geographic polygon data.
""")

st.write("Visualizing the Distribution of Tracks by Country of Origin using a Pie Chart:")

if st.button("Show Country Pie Chart"):
    with st.spinner("Generating pie chart..."):
        country_distribution = merged_df.groupby('Country')['Track'].count()

        fig4, ax4 = matplotlib.pyplot.subplots(figsize=(8, 8))
        
        if not country_distribution.empty:
            ax4.pie(country_distribution, labels=country_distribution.index, autopct='%1.1f%%', startangle=90)
            
            ax4.set_title('Track Distribution by Country of Origin', fontsize=16, fontweight='bold')
            ax4.axis('equal') 
            st.pyplot(fig4)
            st.write("""
The pie chart is divided into 3 parts: USA, UK, and Ireland. However, they are not evenly divided, with most of the bands coming from the USA, a third from the UK, and less than 1% from Ireland. 

The USA is without a doubt the economic powerhouse of the alt rock industry, especially considering that most of the bands that have a high popularity are recording tracks in the country. The UK is the second largest making up for almost the rest of the pie chart, while Ireland makes up a merely 0.06%.
""")

st.write("Visualizing the mapped regions geographically using **GeoPandas**:")

if st.button("Extract Geographic Boundaries"):
    with st.spinner("Extracting and drawing polygons..."):
        
        world = gpd.read_file("https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json")
        
        world['name'] = world['name'].replace({
            'United States of America': 'USA',
            'United Kingdom': 'UK'
        })
        
        alt_rock_countries = world[world['name'].isin(['USA', 'UK', 'Ireland'])]
        
        fig5, ax5 = matplotlib.pyplot.subplots(figsize=(10, 6))
        
        world.plot(ax=ax5, color='#e9ecef', edgecolor='white')
        
        world[world['name'] == 'USA'].plot(ax=ax5, color='red', edgecolor='black')
        world[world['name'] == 'UK'].plot(ax=ax5, color='blue', edgecolor='black')
        world[world['name'] == 'Ireland'].plot(ax=ax5, color='green', edgecolor='black')
        
        ax5.set_title("Geographic Origins of Classic Alt-Rock Artists", fontsize=14, fontweight='bold')
        ax5.axis("off")

        import matplotlib.patches as mpatches
        usa_patch = mpatches.Patch(color='red', label='USA')
        uk_patch = mpatches.Patch(color='blue', label='UK')
        ireland_patch = mpatches.Patch(color='green', label='Ireland')
        ax5.legend(handles=[usa_patch, uk_patch, ireland_patch], loc='lower left', title="Artist Origins")
        
        st.pyplot(fig5)

st.divider()

#X 6. Matplotlib Graphical Representation
st.subheader("6. Graphical Representation (`matplotlib`)")
st.write("""
**Identifying and cleanly representing the top 10 most popular individual alt-rock tracks across the dataset.**
We used matplotlib to create the customized bar chart of the top 10 tracks and the sorting algorithm `.sort_values(by="Popularity", ascending=False)`, after which we isolated the top results using `.head(10)`.

Using the full `matplotlib` package to create a bar chart of the top 10 most popular tracks:
""")

if st.button("Show Matplotlib Chart"):
    with st.spinner("Rendering bar chart..."):
        top_10 = filtered_df.sort_values(by="Popularity", ascending=False).head(10)

        fig2, ax2 = matplotlib.pyplot.subplots(figsize=(12, 6))

        import matplotlib.colors as mcolors

        all_colors_top = list(mcolors.cnames.keys())
        ax2.bar(top_10["Track"], top_10["Popularity"], color=all_colors_top[:len(top_10)], edgecolor="black")

        ax2.set_xticks(range(len(top_10)))
        ax2.set_xticklabels(top_10["Track"], rotation=45, ha='right')

        ax2.set_xlabel("Track Name", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Spotify Popularity Score", fontsize=12, fontweight='bold')
        ax2.set_title("Top 10 Most Popular Alt-Rock Tracks", fontsize=16)

        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig2)
        st.write("""
The bar chart showcases the 10 most popular tracks available in the dataset. It is estimated that these songs alone have generated roughly $74 Million in Spotify royalties. Out of these tracks, a grand total of 9 are coming from the United States (The songs 'Chop Suey!' and 'Toxicity' can be considered Armenian as the band members are of Armenian descent, however we decided to count them as American since the band formed in Los Angeles). The only outlier in this is the British song 'Should I stay or Should I go' from The Clash. This yet again emphasizes the influence of the United States music industry and the large earning gaps between countries and continents around the world.
""")

st.divider()

#X 7. Statistical Modeling: Multiple Regression (`statsmodels`)
st.subheader("7. Statistical Modeling: Multiple Regression (`statsmodels`)")
st.write("""
**Using mathematical modelling to analyse how significant danceability and energy can predict a track's ultimate popularity of a song.**
For this exercise we focused on using the `statsmodels` package to see if these two can predict popularity, we also used streamlit and matplotlib.

Using the mathematical `statsmodels.api` package to analyze how significantly `Danceability` and `Energy` statistically predict a track's ultimate `Popularity` on Spotify:
""")

if st.button("Run Multiple Regression Analysis"):
    with st.spinner("Running OLS mathematical regression..."):
        
        regression_df = filtered_df[['Popularity', 'Danceability', 'Energy']].dropna()
        
        Y = regression_df['Popularity']
        X = regression_df[['Danceability', 'Energy']]
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        st.write("**Comprehensive Statsmodels Regression Summary:**")
        st.text(model.summary().as_text())
        
        st.write("**Actual vs Predicted Popularity Scatter:**")
        predictions = model.predict(X)
        
        fig_reg, ax_reg = matplotlib.pyplot.subplots(figsize=(10, 6))
        ax_reg.scatter(Y, predictions, color='magenta', alpha=0.6, edgecolor='black')
        
        min_val = min(Y.min(), predictions.min())
        max_val = max(Y.max(), predictions.max())
        ax_reg.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
        
        ax_reg.set_xlabel("Actual Algorithm Popularity", fontweight='bold')
        ax_reg.set_ylabel("Predicted Popularity (from audio features)", fontweight='bold')
        ax_reg.set_title("OLS Multiple Regression Accuracy", fontsize=15, fontweight='bold')
        ax_reg.grid(linestyle='--', alpha=0.5)
        
        st.pyplot(fig_reg)
        st.write("""
From the regression results we have:
- The **R-squared value of 0.015** meaning that danceability and energy combined explain only 1.5% of the variance. This means we have an extremely weak model and 98.5% of what makes an alt-rock track popular is driven by variables that are not used in this model.
- A **P-value for the F-statistic of 0.076** meaning that the overall model is not statistically significant.
- Out of danceability and energy, the only statistically significant one is **Energy**. A 1-unit maximum increase in energy is associated with an 11.37-point increase in popularity.

This scatter plot shows us how well our statsmodels regression equation performed. If our model would have been 100% accurate, every single predicted popularity score would have matched the Spotify popularity score. But since our variables are so weakly correlated to the outcome, every single prediction falls between 55 and 65 on the Y-axis.

**In conclusion**, it is better to analyze other factors to determine popularity, especially commercial factors since the industry is heavily driven by marketing budgets.
""")

st.divider()

#X 8. Advanced Encoding: Track Length & Popularity Trends
st.subheader("8. Advanced Encoding: Track Length & Popularity Trends")
st.write("""
**Encoding continuous track lengths into categorical 'Formats' (Short, Standard, Long) to visualize trends over time and calculate economic efficiency.**
We used duration, year, and popularity for this analysis and we made use of streamlit, extreme minimum and maximum values, encoding methods (`pd.get_dummies()`), pandas for calculating the average popularity per year and average return of income per format, and lastly matplotlib for graphical representation.

Encoding track lengths into categorical 'Formats' and visualizing popularity trends sorted by Year.
""")

if st.button("Run Format Analysis"):
    with st.spinner("Processing encoding and sorting by Year..."):
        
        shortest_10 = merged_df.sort_values('Duration').head(10).copy()
        shortest_10['Format'] = 'Short Form (<3.5m)'
        
        longest_10 = merged_df.sort_values('Duration', ascending=False).head(10).copy()
        longest_10['Format'] = 'Long Form (>6m)'
        
        sorted_all = merged_df.sort_values('Duration')
        mid_idx = len(sorted_all) // 2
        middle_10 = sorted_all.iloc[mid_idx-5 : mid_idx+5].copy()
        middle_10['Format'] = 'Standard Form (Mid)'

        adv_df = pd.concat([shortest_10, longest_10, middle_10])
        adv_df = adv_df.dropna(subset=['Year']).sort_values('Year')
        adv_df['Duration_Mins'] = adv_df['Duration'] / 60000

        format_encoded = pd.get_dummies(adv_df['Format'], prefix='Type', dtype=int)
        adv_df = pd.concat([adv_df, format_encoded], axis=1)

        st.write("### The One-Hot Encoded Dataset (Sorted by Year)")
        st.write("We have exactly **10 songs** for each encoded product type (Short, Standard, Long):")
        type_cols = [c for c in adv_df.columns if 'Type_' in c]
        st.dataframe(adv_df[['Year', 'Artist', 'Track', 'Duration_Mins'] + type_cols + ['Popularity']])


        yearly_pop = adv_df.groupby(['Year', 'Format'], observed=True)['Popularity'].mean().unstack()
        fig_final, ax_final = matplotlib.pyplot.subplots(figsize=(12, 6))
        
        for format_type in yearly_pop.columns:
            data = yearly_pop[format_type].dropna()
            ax_final.plot(data.index, data.values, marker='o', markersize=4, label=format_type, alpha=0.8)

        ax_final.set_xlabel("Release Year")
        ax_final.set_ylabel("Average Popularity")
        ax_final.set_title("Market Demand for Balanced Track Formats")
        ax_final.legend()
        st.pyplot(fig_final)

        st.write("""
- This line chart tracks the historical performance of three specific track formats—**Short (<3.5m)**, **Standard (Mid)**, and **Long (>6m)**—across four decades of alt-rock releases, using their modern-day Spotify Popularity scores.
- The **Long Form format (>6m)** is historically the most volatile and generally the lowest-performing category.
- The **Short Form format (<3.5m)** has the highest form ROI, maintaining a stable and high popularity over the years.
- The **Standard Form (Mid-length)** is highly erratic but it also has the absolute highest peaks on the chart.
""")

        st.write("---")
        st.subheader("Economic Interpretation: Efficiency of Attention")
        st.write("We calculate the **Return on Investment (ROI)** as Popularity captured per minute of music across our balanced samples:")

        adv_df['Pop_ROI'] = adv_df['Popularity'] / adv_df.Duration_Mins
        roi_stats = adv_df.groupby('Format', observed=True)['Pop_ROI'].mean()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Short Form ROI", f"{roi_stats['Short Form (<3.5m)']:.2f}", "Pop/Minute")
        with col_b:
            st.metric("Standard Form ROI", f"{roi_stats['Standard Form (Mid)']:.2f}", "Pop/Minute")
        with col_c:
            st.metric("Long Form ROI", f"{roi_stats['Long Form (>6m)']:.2f}", "Pop/Minute")

        st.info("**Statistical Takeaway:** Shorter 'Radio Edit' products almost always capture a higher concentration of popularity per minute. This proves the **Attention Scarcity Theory**: in a digital market, the most efficient financial strategy is producing shorter content that maximizes consumer interest in the smallest possible 'time-window'.")
        
        st.write("""
The Return on Investment results highlights that shorter tracks are doing way better compared to standard and long. The economic efficiency only decays as the track gets longer and this proves that in the digital era of consumerism attention span is vital and usually shorter forms of media are more likely to be engaged with by the consumers.
""")

st.divider()

#X 9. Data Normalization: Scaling Methods
st.subheader("9. Data Normalization: Scaling Methods")
st.write("""
**Using `sklearn.preprocessing.StandardScaler` to translate raw musical data measured with different units into a single Standardized Market Index to measure them accordingly.**
To do this we utilized z-score standardization to scale tempo, energy and popularity in a standardized format, allowing us to eliminate the scale-bias. We made use of applying the `sklearn.preprocessing.StandardScaler()` algorithm and the mathematical formula $z = (x - \\mu) / \\sigma$ (where $x$ = raw data, $\\mu$ = mean of the feature, $\\sigma$ = standard deviation).

Using `sklearn.preprocessing.StandardScaler` to translate musical features with different units into a single **Standardized Market Index**.
""")

if st.button("Calculate Unified Scaling Model"):
    with st.spinner("Standardizing disparate musical assets..."):
        scale_data = filtered_df[['Popularity', 'Tempo', 'Energy']].dropna().copy()
        
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(scale_data)
        
        scaled_df = pd.DataFrame(scaled_array, columns=['Scaled_Popularity', 'Scaled_Tempo', 'Scaled_Energy'])
        
        st.write("Notice how every musical feature now has a **Mean of 0** and a **Standard Deviation of 1**. They can now be mathematically compared without scale-bias:")
        st.dataframe(scaled_df.head(10))
        
        fig_scale, (ax_pre, ax_post) = matplotlib.pyplot.subplots(1, 2, figsize=(14, 6))
        
        scale_data.boxplot(ax=ax_pre)
        ax_pre.set_title("1. Pre-Scaling: Disparate Units (Market Chaos)")
        ax_pre.set_ylabel("Original Raw Units (0 to 180+)")
        
        scaled_df.boxplot(ax=ax_post)
        ax_post.set_title("2. Post-Scaling: Standardized Units (Unified Matrix)")
        ax_post.set_ylabel("Standard Deviations (σ)")
        
        st.pyplot(fig_scale)
        
        st.write("""
- In the first boxplot, **tempo** appears to be far more valuable than energy because they are measured in different ways and operate on different numerical scales. Tempo is measured in BPM (beats per minute) and the range is between 60 to 100, while energy is a ratio between 0.0 and 1.0.
- In the second boxplot, we can visualize how **z-score standardization** eliminates the bias created by larger numbers. Now energy is no longer flattened at the bottom of the chart and we can see that it actually contains a dynamic distribution of data.

This shows how in an efficient market value cannot be measured in raw units and in order to have a better understanding of a track's true market potential, these metrics must be placed on an equal mathematical playing field.
""")
        
        st.success("**Economic Interpretation:** In an efficient market, 'Value' cannot be measured in raw units. For example, earning **1 more point of Popularity** is extremely difficult (Scarcity), while adding **1 BPM to Tempo** is easy. By scaling everything to the same **Z-score scale**, we create a 'Unified Asset Index' where different features can finally be added together with equal mathematical weight.")
