import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

st.set_page_config(page_title='Ch. 8 - Measurement Error', layout="wide")
st.title('Ch. 8 - Measurement Error')

st.markdown(
'''
This dashboard explores the impact of measurement error on regression analysis.
You can simulate different types of measurement errors in the distance and price variables, visualize their distributions, and see how they affect regression results.
The dashboard allows you to adjust parameters such as error type, standard deviation, mean, and correlation with the rating variable.
You can also choose to log-transform the variables and select the polynomial degree for regression.
The dashboard is built on the `hotels-europe` dataset, focusing on hotel prices in London on weekdays in November 2017 (N = 578).
The data has been filtered to include only hotels with at least 3 stars, within 10 miles of the city center, and with a valid rating.
''')

# Load and cach data from OSF
@st.cache_data
def load_data():
    hotels_price = pd.read_csv('data/hotels-europe_price.csv')
    hotels_features = pd.read_csv('data/hotels-europe_features.csv')
    data = pd.merge(hotels_features[
                (hotels_features['city'] == 'London') &
                (hotels_features['accommodation_type'] == 'Hotel') &
                (hotels_features['stars'] >= 3) &
                (hotels_features['distance'] <= 10) &
                (hotels_features['rating'].notna())
            ], hotels_price[
                (hotels_price['year'] == 2017) &
                (hotels_price['month'] == 11) &
                (hotels_price['weekend'] == 0)
            ], on='hotel_id', how='inner')
    data = data[['hotel_id', 'distance', 'price', 'rating']]
    return data

data = load_data()

# Sidebar for settings
st.sidebar.header("Measurement Error Settings")
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=10000, value=42)
np.random.seed(seed)

def measurement_error_input(label, mean_min, mean_max, mean_def, sd_min, sd_max, sd_def):
    error_type = st.sidebar.selectbox(f"{label} Error Type", ["None", "Classical", "Non-Classical (Bias)", "Non-Classical (Correlated)"],
                                      help='''
* Classical: Adds normally distributed error with mean 0 and specified SD.
* Non-Classical (Bias): Adds normally distributed error with specified mean and SD.
* Non-Classical (Correlated): Adds error correlated with rating.
''')
    if error_type == "Classical":
        sd = st.sidebar.slider(f"{label} Error SD", sd_min, sd_max, sd_def)
        return lambda df: np.random.normal(0, sd, size=len(df))
    elif error_type == "Non-Classical (Bias)":
        mean = st.sidebar.slider(f"{label} Error Mean", mean_min, mean_max, mean_def)
        sd = st.sidebar.slider(f"{label} Error SD", sd_min, sd_max, sd_def)
        return lambda df: np.random.normal(mean, sd, size=len(df))
    elif error_type == "Non-Classical (Correlated)":
        base_mean = st.sidebar.slider(f"{label} Base Mean", mean_min, mean_max, mean_def)
        base_sd = st.sidebar.slider(f"{label} Base SD", sd_min, sd_max, sd_def)
        corr = st.sidebar.slider(f"{label} Error Correlation with Rating", -1.0, 1.0, 0.0)
        return lambda df: base_mean + corr * df['rating'].values + np.random.normal(0, base_sd, size=len(df))
    else:
        return lambda df: np.zeros(len(df))

x_error_fn = measurement_error_input("X (Distance)", -5.0, 5.0, 0.0, 0.1, 3.0, 0.1)
y_error_fn = measurement_error_input("Y (Price)", -50.0, 50.0, 0.0, 0.1, 100.0, 10.0)

# Log transformation and model degree
st.sidebar.header("Model Settings")
log_x = st.sidebar.checkbox("Log-transform X (Distance)")
log_y = st.sidebar.checkbox("Log-transform Y (Price)")

# Filter out less than 1 mile
st.sidebar.header("Data Filtering")
filtersmall = st.sidebar.checkbox("Filter out hotels with distance < 1 mile", value=True)
if filtersmall:
    data = data[data['distance'] >= 1]

# Create noisy data
x = data['distance'].copy()
y = data['price'].copy()

x_err = x + x_error_fn(data)
y_err = y + y_error_fn(data)

x_err[x_err <= 0] = x.min()
y_err[y_err <= 0] = y.min()

# Apply log transformations if selected
if log_x:
    x_plot = np.log(x)
    x_err_plot = np.log(x_err)
else:
    x_plot = x
    x_err_plot = x_err

if log_y:
    y_plot = np.log(y)
    y_err_plot = np.log(y_err)
else:
    y_plot = y
    y_err_plot = y_err

# Histogram plots with overlays
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Distance histogram overlay
sns.histplot(x_plot, ax=axs[0], color=color[0], alpha=0.5, label="Without Error", binwidth=0.5 if not log_x else 0.25)
sns.histplot(x_err_plot, ax=axs[0], color=color[1], alpha=0.5, label="With Error", binwidth=0.5 if not log_x else 0.25)
axs[0].set_xlabel("Distance" if not log_x else "ln(Distance)")
axs[0].set_ylabel("Frequency")
axs[0].spines[['top', 'right']].set_visible(False)
axs[0].legend()

# Price histogram overlay
sns.histplot(y_plot, ax=axs[1], color=color[0], alpha=0.5, label="Without Error", binwidth=25 if not log_y else 0.25)
sns.histplot(y_err_plot, ax=axs[1], color=color[1], alpha=0.5, label="With Error", binwidth=25 if not log_y else 0.25)
axs[1].set_xlabel("Price" if not log_y else "ln(Price)")
axs[1].set_ylabel("Frequency")
axs[1].spines[['top', 'right']].set_visible(False)
axs[1].legend()

# Show the plots
st.subheader("Distribution of Variables with and without Measurement Error")
st.write('Note: In the data with measurement error, non-positive values have been replaced with the minimum of the original data to avoid meaningless values.')
st.pyplot(fig, clear_figure=False)

# Combined regression plot
fig, ax = plt.subplots(figsize=(5, 3))

# Scatter: without error
ax.scatter(x_plot, y_plot, color=color[0], alpha=0.25, s=10, edgecolor='none')
model_raw = sm.OLS(y_plot, np.vander(x_plot, N=2, increasing=True)).fit(cov_type='HC3')
x_fit = np.linspace(min(x_plot), max(x_plot), 100)
y_fit = model_raw.predict(np.vander(x_fit, N=2, increasing=True))
ax.plot(x_fit, y_fit, color=color[0], linewidth=2, label="Without Error")

# Scatter: with error
ax.scatter(x_err_plot, y_err_plot, color=color[1], alpha=0.25, s=10, edgecolor='none')
model_err = sm.OLS(y_err_plot, np.vander(x_err_plot, N=2, increasing=True)).fit(cov_type='HC3')
x_fit_err = np.linspace(min(x_err_plot), max(x_err_plot), 100)
y_fit_err = model_err.predict(np.vander(x_fit_err, N=2, increasing=True))
ax.plot(x_fit_err, y_fit_err, color=color[1], linewidth=2, label="With Error")

# Labels and styling
ax.set_xlabel("ln(Distance)" if log_x else "Distance")
ax.set_ylabel("ln(Price)" if log_y else "Price")
ax.spines[['top', 'right']].set_visible(False)
ax.legend()

st.subheader("Regression with and without Measurement Error")
st.pyplot(fig, use_container_width=False)

# Construct table with regression results
def stargazer_table(model, label_x):
    coefs = model.params
    ses = model.bse
    pvals = model.pvalues

    def stars(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.1:
            return '*'
        else:
            return ''

    rows = []
    suffixes = ['', ' sq.', ' cub.']
    for i in range(len(coefs)):
        if i == 0:
            name = "Intercept"
        else:
            name = f"{label_x}{suffixes[i-1]}"
        coef_str = f"{coefs[i]:.2f}{stars(pvals[i])}"
        se_str = f"({ses[i]:.2f})"
        rows.append([name, coef_str, se_str])

    r2 = model.rsquared
    rows.append(["R-squared", f"{r2:.2f}", ""])

    df = pd.DataFrame(rows, columns=["Variable", "Coefficient", "Standard Error"])
    return df.set_index("Variable")

# Combine tables with multi-index columns
label_x = "Distance" if not log_x else "ln(Distance)"
table_raw = stargazer_table(model_raw, label_x)
table_err = stargazer_table(model_err, label_x)

merged = pd.concat([table_raw, table_err], axis=1, keys=["Without Measurement Error", "With Measurement Error"])

# Display regression results
st.subheader("Comparison of Regression Outputs")
st.dataframe(merged)
st.markdown('*Note: Robust standard errors (HC3) used. *** significant at 1%, ** significant at 5%, * significant at 10%.*')