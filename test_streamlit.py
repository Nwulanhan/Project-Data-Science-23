import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# Set page title and favicon
st.set_page_config(page_title="World Economic Dashboard 🌍", page_icon=":globe_with_meridians:", layout="wide")

# Custom CSS styles
st.markdown(
    """
    <style>
    .reportview-container {
        background-image: url('https://images.unsplash.com/photo-1523961131990-5ea7c61b2107?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80');
        background-size: cover;
        background-position: center;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stNumberInput > div > input,
    .stSelectbox > div > div {
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
    .stContainer {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    .stTextInput > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        transition: border-color 0.3s;
    }
    .stTextInput > div > input:focus,
    .stNumberInput > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #66bb6a;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load data using st.cache_data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(url, sheet_name):
    try:
        df = pd.read_excel(url, sheet_name=sheet_name, engine="openpyxl")
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].astype(str).str.strip()
                df[column] = df[column].str.replace(',', '')
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# URLs for the datasets
url_factbook = "https://github.com/Nwulanhan/factbookPSD/raw/main/factbookproyek.xlsx"
url_weo = "https://github.com/Nwulanhan/factbookPSD/raw/main/WEOApr2024all.xlsx"

# Load datasets
try:
    df_factbook = load_data(url_factbook, 'factbook (2)')
    df_weo = load_data(url_weo, 'WEOApr2024all')
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar menu with icons
with st.sidebar:
    selected = option_menu("Menu", ["Visualisasi", 'Prediksi'], icons=['bar-chart', 'search'], menu_icon="house", default_index=0)

if df_weo is not None and selected == "Visualisasi":
    st.title('Dashboard Ekonomi Dunia 🌍')

    try:
        # Data processing and cleaning
        columns_to_clean = df_weo.columns[10:]
        df_weo[columns_to_clean] = df_weo[columns_to_clean].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
        df_ngdp = df_weo[(df_weo['WEO Subject Code'] == 'NGDP')]

        # Container for the choropleth map
        with st.container():
            st.subheader("Persebaran GDP di Dunia 🗺️")
            year_columns = [col for col in df_ngdp.columns[10:] if str(col).isdigit()]
            year_columns = list(map(int, year_columns))
            selected_year = st.slider('Pilih Tahun 📅', min_value=min(year_columns), max_value=max(year_columns), step=1)
            year_column = int(selected_year)

            df_weo_year = df_ngdp[['Country', 'Continent', year_column]].dropna()
            df_weo_year.columns = ['Country', 'Continent', 'GDP']

            fig_choropleth = px.choropleth(df_weo_year,
                                           locations="Country",
                                           locationmode='country names',
                                           color="GDP",
                                           hover_name="Country",
                                           hover_data=["Country", "Continent", "GDP"],
                                           title=f"Persebaran GDP tahun {year_column} 🗺️",
                                           color_continuous_scale=px.colors.sequential.Magenta)
            fig_choropleth.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'))
            st.plotly_chart(fig_choropleth)

        # Container for the top 10 countries bar chart
        with st.container():
            st.subheader("Top 10 Negara dengan GDP Tertinggi 🏆")
            top_10_countries = df_weo_year.sort_values(by='GDP', ascending=False).head(10)

            fig_bar_horizontal = go.Figure()
            fig_bar_horizontal.add_trace(
                go.Bar(
                    y=top_10_countries['Country'],
                    x=top_10_countries['GDP'],
                    orientation='h',
                    text=top_10_countries.apply(lambda row: f"{row['Continent']}, {row['Country']}: {row['GDP']}", axis=1),
                    hoverinfo='text',
                    marker=dict(color='slateblue'),
                )
            )
            fig_bar_horizontal.update_layout(
                title=f"Top 10 Negara dengan GDP Tertinggi tahun {year_column}",
                xaxis_title="GDP",
                yaxis_title="Country",
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_bar_horizontal)

        # Container for the line chart
        with st.container():
            st.subheader("Perkembangan GDP per Benua 📈")
            continent = st.selectbox('Pilih Benua 🌎', ['Asia', 'Europe', 'North America', 'South America', 'Oceania'])

            df_ngdp = df_weo[(df_weo['WEO Subject Code'] == 'NGDP') & (df_weo['Continent'] == continent)].set_index('Country')
            years = df_ngdp.columns[10:-1]
            df_ngdp = df_ngdp[years]
            df_ngdp.columns = df_ngdp.columns.astype(int)
            df_ngdp = df_ngdp.dropna(axis=1, how='all')

            countries = df_weo[(df_weo['WEO Subject Code'] == 'NGDP') & (df_weo['Continent'] == continent)]['Country'].unique()
            selected_countries = st.multiselect('Pilih Negara 🌍', countries)

            fig = go.Figure()
            for country in selected_countries:
                fig.add_trace(go.Scatter(x=df_ngdp.columns, y=df_ngdp.loc[country],
                                        mode='lines+markers',
                                        name=country))
            fig.update_layout(title=f'Perkembangan GDP Negara-negara {continent} 📈',
                            xaxis_title='Tahun',
                            yaxis_title='GDP (dalam mata uang nasional)',
                            legend_title='Negara',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            hovermode='x unified')
            st.plotly_chart(fig)

    except KeyError as e:
        st.error(f'Kolom yang hilang atau format data tidak sesuai untuk time series: {str(e)}')

if df_factbook is not None and selected == "Prediksi":
    st.title('Prediksi GDP 🌍💰')

    try:
        # Extracting features and target variable
        x = df_factbook[['Area', 'Electricity consumption', 'Population ']]
        y = df_factbook['GDP ']

        # Splitting the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Initialize and train the SGD Regressor model
        sgdr = SGDRegressor(random_state=42)
        sgdr.fit(x_train_scaled, y_train)

        # Container for user input
        with st.container():
            st.subheader("Prediksi GDP berdasarkan Masukan Berikut")

            area = st.number_input("Luas Wilayah (kilometer persegi) 🗺️", min_value=0.30, value=0.30, step=0.1)
            electricity_consumption = st.number_input("Konsumsi Listrik (kWh per kapita) 💡", min_value=1000.0, value=1000.0, step=0.10)
            population = st.number_input("Populasi 👥", min_value=500.0, value=500.0, step=50.0)
            country_name = st.text_input("Nama Negaramu 🌍")

            if st.button("Prediksi GDP 🔮"):
                input_data = pd.DataFrame({'Area': [area],
                                           'Electricity consumption': [electricity_consumption],
                                           'Population ': [population]})
                input_data_scaled = scaler.transform(input_data)
                predicted_gdp = sgdr.predict(input_data_scaled)
                st.success(f"Prediksi GDP untuk negara {country_name}: ${predicted_gdp[0]:,.2f} 💰")

    except KeyError as e:
        st.error(f'Kolom yang hilang atau format data tidak sesuai: {str(e)}')
