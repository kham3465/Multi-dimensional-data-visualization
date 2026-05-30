"""
Script to generate multi‑dimensional data visualizations for the
CMP_SC‑8630 data visualization assignment.  The script loads three
real‑world datasets related to climate and hydrology and produces
visualizations that explore patterns across multiple variables and
dimensions.  The resulting figures are saved to the ``output``
directory.  The datasets used here include:

* ``weather_data.csv`` – daily weather observations for multiple
  cities in New Zealand (2016–2017) containing temperature,
  humidity, wind, pressure and precipitation variables.  Source:
  mosaicData package within the Rdatasets collection.
* ``global_temp.csv`` – NASA Goddard Institute for Space Studies
  (GISTEMP) global land–ocean temperature anomalies from 1880 to
  2025.  Monthly anomalies relative to the 1951–1980 baseline are
  provided.  Source: NASA GISS via data.giss.nasa.gov.
* ``minnesota_weather.csv`` – monthly weather summary for six
  Minnesota agricultural sites (1927–1936) including cooling and
  heating degree days, precipitation and temperature extremes.
  Source: agridat package within Rdatasets.

The visualizations include heatmaps, scatter plots and line charts
to illustrate how variables such as temperature, humidity and
precipitation vary over time and across different locations.
"""

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D


def ensure_output_dir(path: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_weather_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of average temperature by city and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Weather data with columns ``city``, ``month`` and ``avg_temp``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # 1. Compute the average monthly temperature for each city
    avg_monthly = df.groupby(['city', 'month'])['avg_temp'].mean().reset_index()
    
    # 2. & 3. Pivot and sort columns
    pivot_df = avg_monthly.pivot(index='city', columns='month', values='avg_temp')
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    # 4. Create a heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.1f', 
                cbar_kws={'label': "Average temperature"})
    
    # 5. Set labels
    plt.title("Average monthly temperature by city")
    plt.xlabel("Month")
    plt.ylabel("City")

    # 6. Save figure
    out_path = os.path.join(outdir, "weather_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_weather_scatter(df: pd.DataFrame, outdir: str) -> str:
    """Create a scatter plot exploring relationships between humidity,
    temperature and precipitation.

    Each point represents a daily observation.  The x‑axis shows
    average humidity, the y‑axis shows average temperature in Fahrenheit,
    the marker size encodes precipitation and colour encodes the city.
    Separate legends are provided for city and precipitation to avoid
    overlap.

    Parameters
    ----------
    df : pandas.DataFrame
        Weather data with columns ``avg_humidity``, ``avg_temp``,
        ``precip`` and ``city``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # 1. Clean the 'precip' column
    df = df.copy()
    df['precip'] = pd.to_numeric(df['precip'], errors='coerce').fillna(0.0)
    
    # 2. & 3. Set up figure and size range
    plt.figure(figsize=(9, 6))
    size_range = (20, 300)
    
    # 4. Generate scatter plot
    palette = sns.color_palette("husl", len(df['city'].unique()))
    ax = sns.scatterplot(data=df, x="avg_humidity", y="avg_temp", hue="city", 
                         size="precip", sizes=size_range, alpha=0.65, 
                         legend=False, palette=palette)

    # 5. Custom legend for cities
    cities = sorted(df['city'].unique())
    city_handles = [Line2D([0], [0], marker='o', color='w', label=city,
                           markerfacecolor=palette[i], markersize=10) 
                    for i, city in enumerate(cities)]
    city_legend = plt.legend(handles=city_handles, loc="upper left", 
                             bbox_to_anchor=(1.02, 1.0), title="City")
    ax.add_artist(city_legend)

    # 6. Custom legend for precipitation
    precip_vals = [0, df['precip'].max() * 0.2, df['precip'].max() * 0.5, df['precip'].max()]
    mapped_sizes = np.interp(precip_vals, [0, df['precip'].max()], size_range)
    precip_handles = [plt.scatter([], [], s=s, color='gray', alpha=0.6) for s in mapped_sizes]
    plt.legend(handles=precip_handles, labels=[f"{v:.1f}" for v in precip_vals],
               loc="lower left", bbox_to_anchor=(1.02, 0.0), title="Precipitation")

    plt.xlabel("Average relative humidity (%)")
    plt.ylabel("Average temperature (°F)")
    plt.title("Daily weather: temperature vs humidity with precipitation (size)")
    
    out_path = os.path.join(outdir, "weather_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_global_temp_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of global temperature anomalies by year and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Global temperature anomalies where rows correspond to years and
        columns to months (Jan–Dec).  The DataFrame should include
        numeric values for anomalies.  Missing values are allowed and
        will appear as blank cells.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # 1. Reshape from wide to long
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    long_df = pd.melt(df, id_vars=['Year'], value_vars=months, var_name="Month", value_name="Anomaly")
    
    # 2. Map Month to numbers
    month_map = {m: i+1 for i, m in enumerate(months)}
    long_df['MonthNum'] = long_df['Month'].map(month_map)
    
    # 3. & 4. Pivot and sort
    pivot_df = long_df.pivot(index='Year', columns='MonthNum', values='Anomaly').sort_index()

    # 5. & 6. Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_df, cmap='coolwarm', vmin=-1.5, vmax=1.5,
                     cbar_kws={'label': "Temperature anomaly (°C relative to 1951–1980)"},
                     linewidths=0, linecolor="white")
    
    # 7. & 8. Customize ticks and titles
    plt.xticks(np.arange(len(months)) + 0.5, months, rotation=45)
    plt.title("Global land–ocean temperature anomalies (1880–2025)")
    plt.xlabel("Month")
    plt.ylabel("Year")

    out_path = os.path.join(outdir, "global_temp_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_minnesota_precip_line(df: pd.DataFrame, outdir: str) -> str:
    """Create a line chart of monthly precipitation by site over time.

    This figure shows how precipitation varies across the six Minnesota
    sites from 1927 to 1936.  Each line corresponds to a site and
    month; values are aggregated by year and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Minnesota weather data with columns ``site``, ``year``, ``mo`` (month) and
        ``precip``.
    outdir : str
        Directory to write the output image.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # 1. Create datetime column
    df = df.copy()
    df['date'] = pd.to_datetime({'year': df['year'], 'month': df['mo'], 'day': 1})
    
    # 2. & 3. Plot line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="date", y="precip", hue="site")
    
    # 4. & 5. Set title and legend
    plt.title("Monthly precipitation by Minnesota site (1927–1936)")
    plt.xlabel("Year")
    plt.ylabel("Precipitation (inches)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Site")

    out_path = os.path.join(outdir, "minnesota_precip_line.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    # Load and plot weather data
    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    # Plot heatmap and scatter
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    # Load and plot global temperature anomalies
    global_path = os.path.join(data_dir, "global_temp.csv")
    global_df = pd.read_csv(global_path, skiprows=1)
    # Replace *** with NA and convert to numeric
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    # Load and plot Minnesota weather data
    minn_path = os.path.join(data_dir, "minnesota_weather.csv")
    minn_df = pd.read_csv(minn_path)
    figures.append(plot_minnesota_precip_line(minn_df, out_dir))
    return figures


if __name__ == "__main__":
    generated = main()
    print("Generated figures:")
    for path in generated:
        print(path)