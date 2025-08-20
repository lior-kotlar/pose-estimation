
import os
import numpy as np
import h5py
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extract_flight_data import FlightAnalysis
from extract_flight_data import save_movies_data_to_hdf5

# matplotlib.use('TkAgg')


def collect_full_body_wingbits(base_dir, dir_label):
    full_body_wingbits = {}

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("mov") and file.endswith("_analysis_smoothed.h5"):
                print(f"file: {file} dir: {dir_label}")
                file_path = os.path.join(root, file)
                with h5py.File(file_path, 'r') as h5_file:
                    if "full_body_wing_bits" in h5_file:
                        wing_bits_group = h5_file["full_body_wing_bits"]
                        for wingbit_key in wing_bits_group:
                            wingbit_group = wing_bits_group[wingbit_key]
                            unique_group_name = f"{dir_label}_{file}_{wingbit_key}"
                            if unique_group_name not in full_body_wingbits:
                                full_body_wingbits[unique_group_name] = {}
                            for dataset_key in wingbit_group:
                                full_body_wingbits[unique_group_name][dataset_key] = wingbit_group[dataset_key][...]

    return full_body_wingbits


def combine_wingbits_and_save(output_file, *directories):
    combined_wingbits = {}

    # Collect data from each directory
    for i, directory in enumerate(directories):
        full_body_wingbits = collect_full_body_wingbits(directory, f'dir{i + 1}')
        combined_wingbits.update(full_body_wingbits)

    print(f"saving data to {output_file}")
    with h5py.File(output_file, 'w') as h5_output_file:
        full_body_wing_bits_grp = h5_output_file.create_group("full_body_wing_bits")
        for group_name, datasets in combined_wingbits.items():
            wingbit_grp = full_body_wing_bits_grp.create_group(group_name)
            for dataset_key, data in datasets.items():
                wingbit_grp.create_dataset(dataset_key, data=data)


def extract_numbers(input_string):
    # The regex pattern
    pattern = r"dir(\d+)_mov(\d+)_.*_wingbit_(\d+)"

    # Perform the regex search
    match = re.search(pattern, input_string)

    if match:
        dir_number = match.group(1)
        mov_number = match.group(2)
        wingbit_number = match.group(3)
        return dir_number, mov_number, wingbit_number
    else:
        return None, None, None


def read_h5_file_and_collect_take_attributes(file_path):
    collected_data = []
    with h5py.File(file_path, 'r') as h5_file:
        full_body_wing_bits_grp = h5_file['full_body_wing_bits']
        for i, wingbit_key in enumerate(full_body_wing_bits_grp):
            print(f"{wingbit_key}")
            wingbit_grp = full_body_wing_bits_grp[wingbit_key]
            dir_number, mov_number, wingbit_number = extract_numbers(input_string=wingbit_key)
            row_data = {'wingbit': f"dir{dir_number}_mov{mov_number}_wingbit_{wingbit_number}"}
            for dataset_key in wingbit_grp:
                if 'take' in dataset_key:
                    data = wingbit_grp[dataset_key][()]
                    row_data[dataset_key] = data
            collected_data.append(row_data)
    return collected_data


def create_dataframe(collected_data):
    return pd.DataFrame(collected_data)


def visualize_dataframe(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.set_index('wingbit').transpose(), annot=True, cmap='coolwarm', cbar=True)
    plt.title('Mean Attributes for Each Wingbit')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def load_dataframe_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def sort_wingbits_by_movie(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract the wingbit number from the wingbit column
    df['wingbit_num'] = df['wingbit'].str.extract(r'(\d+)$').astype(int)

    # Extract the movie identifier from the wingbit column
    df['movie'] = df['wingbit'].str.extract(r'^(.*)_wingbit_\d+$')

    # Sort the dataframe by movie and wingbit_num
    df_sorted = df.sort_values(by=['movie', 'wingbit_num'])

    # Drop the temporary columns
    df_sorted = df_sorted.drop(columns=['wingbit_num', 'movie'])

    # Save the sorted dataframe back to the same file
    df_sorted.to_csv(file_path, index=False)


def create_dataframe_from_h5(h5_file_path, file_name="all_wingbits_attributes.csv"):
    # Check if file exists
    print(f"the file is {h5_file_path}", flush=True)
    if not os.path.exists(h5_file_path):
        print(f"File not found: {h5_file_path}")
        return

    # Collect data from the HDF5 file
    collected_data = read_h5_file_and_collect_take_attributes(h5_file_path)

    # Create a DataFrame from the collected data
    df = create_dataframe(collected_data)
    print(f"the data frame is {df}", flush=True)
    dir_path = os.path.dirname(h5_file_path)
    # dir_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data"
    csv_path = os.path.join(dir_path, file_name)
    save_dataframe_as_csv(df, csv_path)
    sort_wingbits_by_movie(csv_path)
    return df


def confidence_interval(r, n, alpha=0.05):
    # Fisher Z-transformation
    Z = np.arctanh(r)

    # Standard error
    SE_Z = 1 / np.sqrt(n - 3)

    # Confidence interval in Z-space
    Z_critical = stats.norm.ppf(1 - alpha / 2)
    Z_lower = Z - Z_critical * SE_Z
    Z_upper = Z + Z_critical * SE_Z

    # Inverse Fisher Z-transformation
    r_lower = np.tanh(Z_lower)
    r_upper = np.tanh(Z_upper)

    return r_lower, r_upper


def compute_correlations(csv_path, save_name="correlations.html"):
    print(f"computes correlations\n{csv_path}")
    df = pd.read_csv(csv_path)  # Assume load_dataframe_from_csv is equivalent to pd.read_csv

    # Get the size of the data
    num_rows, num_cols = df.shape

    # Drop the first column which is the string column and the specified columns p, q, r
    df_numeric = df.drop(columns=['wingbit'])
    # df_numeric = df

    # Compute Pearson Correlation
    print(df_numeric.dtypes)
    pearson_corr = df_numeric.corr(method='pearson')

    # Compute Spearman Correlation
    spearman_corr = df_numeric.corr(method='spearman')

    # Calculate mean and std for the numerical columns, rounded to 2 significant digits
    stats_df = df_numeric.describe().loc[['mean', 'std']].T.reset_index()
    stats_df['mean'] = stats_df['mean'].round(2)
    stats_df['std'] = stats_df['std'].round(2)
    stats_df.columns = ['Variable', 'Mean', 'Standard Deviation']

    # Remove 'mean' and 'meen' from variable names for display
    display_columns = [col.replace('_take', '').strip() for col in df_numeric.columns]

    # Compute confidence intervals for Pearson and Spearman correlations
    pearson_confidence_intervals = np.zeros((len(pearson_corr), len(pearson_corr), 2))
    spearman_confidence_intervals = np.zeros((len(spearman_corr), len(spearman_corr), 2))
    for i in range(len(pearson_corr)):
        for j in range(len(pearson_corr)):
            r_pearson = pearson_corr.iloc[i, j]
            r_spearman = spearman_corr.iloc[i, j]
            pearson_confidence_intervals[i, j] = confidence_interval(r_pearson, num_rows)
            spearman_confidence_intervals[i, j] = confidence_interval(r_spearman, num_rows)

    # Create a subplot figure with Plotly
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Pearson Correlation', 'Spearman Correlation', 'Statistics'),
        specs=[[{"type": "heatmap"}], [{"type": "heatmap"}], [{"type": "table"}]]
    )

    # Format hover text for Pearson correlation
    pearson_hover_text = [[
        (f"<b>{display_columns[i]}</b> vs <b>{display_columns[j]}</b><br>Correlation: "
         f"{pearson_corr.iloc[i, j]:.2f}<br>95% CI: [{pearson_confidence_intervals[i, j, 0]:.2f}, "
         f"{pearson_confidence_intervals[i, j, 1]:.2f}]")
        for j in range(len(pearson_corr))] for i in range(len(pearson_corr))]

    # Pearson Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pearson_corr.values,
            x=display_columns,
            y=display_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=1, col=1
    )

    # Format hover text for Spearman correlation
    spearman_hover_text = [[
        f"<b>{display_columns[i]}</b> vs <b>{display_columns[j]}</b><br>Correlation: {spearman_corr.iloc[i, j]:.2f}<br>95% CI: [{spearman_confidence_intervals[i, j, 0]:.2f}, {spearman_confidence_intervals[i, j, 1]:.2f}]"
        for j in range(len(spearman_corr))] for i in range(len(spearman_corr))]

    # Spearman Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=spearman_corr.values,
            x=display_columns,
            y=display_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=2, col=1
    )

    # Adding the statistics table
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[stats_df[col] for col in stats_df.columns],
                       fill_color='lavender',
                       align='left')
        ),
        row=3, col=1
    )

    fig.update_layout(height=1500, width=1000,
                      title_text=f'{save_name[:-5]} (number of wingbits: {num_rows}, {num_cols} attributes)')
    save_path = os.path.join(os.path.dirname(csv_path), save_name)
    fig.write_html(save_path)

    print(f'Correlation heatmaps and statistics table saved to {save_path}')


def create_histograms(csv_path, save_name="histograms.html"):
    print(f"Creating histograms\n{csv_path}")
    df = pd.read_csv(csv_path)

    # Drop the first column which is the string column and the specified columns p, q, r
    df_numeric = df.drop(columns=['wingbit'])

    # Calculate mean and std for the numerical columns, rounded to 2 significant digits
    stats = df_numeric.describe().loc[['mean', 'std']].T.reset_index()
    stats['mean'] = stats['mean'].round(2)
    stats['std'] = stats['std'].round(2)
    stats.columns = ['Variable', 'Mean', 'Standard Deviation']

    # Create a subplot figure with Plotly
    fig = make_subplots(rows=len(df_numeric.columns), cols=1,
                        subplot_titles=[
                            f"{col} (Mean: {stats.loc[stats['Variable'] == col, 'Mean'].values[0] if not stats.loc[stats['Variable'] == col, 'Mean'].empty else 'N/A'}, "
                            f"Std: {stats.loc[stats['Variable'] == col, 'Standard Deviation'].values[0] if not stats.loc[stats['Variable'] == col, 'Standard Deviation'].empty else 'N/A'})"
                            for col in df_numeric.columns])

    # Adding histograms
    for i, col in enumerate(df_numeric.columns):
        if df_numeric[col].nunique() <= 1 or df_numeric[col].isnull().all():
            fig.add_trace(
                go.Histogram(
                    x=[],
                    nbinsx=50,
                    marker=dict(color='blue', line=dict(color='black', width=1)),
                    opacity=0.75
                ),
                row=i + 1, col=1
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=df_numeric[col],
                    nbinsx=50,
                    marker=dict(color='blue', line=dict(color='black', width=1)),
                    opacity=0.75
                ),
                row=i + 1, col=1
            )

    fig.update_layout(height=300 * len(df_numeric.columns), width=1000,
                      title_text='Histograms of Data Variables')
    save_path = os.path.join(os.path.dirname(csv_path), save_name)
    fig.write_html(save_path)

    print(f'Histograms saved to {save_path}')


def compute_correlations_from_df(df):
    # Drop non-numeric columns and specified columns
    df_numeric = df.drop(columns=['wingbit', 'mean_p', 'mean_q', 'mean_r', 'mean_body_speed'])
    df_numeric.columns = [col.replace('mean_', '').strip() for col in df_numeric.columns]

    # Compute Pearson and Spearman Correlations
    pearson_corr = df_numeric.corr(method='pearson')
    spearman_corr = df_numeric.corr(method='spearman')

    # Calculate mean and std for the numerical columns, rounded to 2 significant digits
    stats = df_numeric.describe().loc[['mean', 'std']].T.reset_index()
    stats['mean'] = stats['mean'].round(2)
    stats['std'] = stats['std'].round(2)
    stats.columns = ['Variable', 'Mean', 'Standard Deviation']

    display_columns = df_numeric.columns

    return pearson_corr, spearman_corr, stats, display_columns, df_numeric


def display_differences(csv_path1, csv_path2, output_file):
    df1 = load_dataframe_from_csv(csv_path1)
    df2 = load_dataframe_from_csv(csv_path2)
    pearson_corr1, spearman_corr1, stats1, display_columns1, df_numeric1 = compute_correlations_from_df(df1)
    pearson_corr2, spearman_corr2, stats2, display_columns2, df_numeric2 = compute_correlations_from_df(df2)

    # Print columns for debugging
    print("Columns in experiment 1:", display_columns1)
    print("Columns in experiment 2:", display_columns2)

    # Compute the differences
    pearson_diff = pearson_corr2 - pearson_corr1
    spearman_diff = spearman_corr2 - spearman_corr1

    print("Pearson Correlation 1:\n", pearson_corr1)
    print("Pearson Correlation 2:\n", pearson_corr2)
    print("Pearson Correlation Difference:\n", pearson_diff)

    print("Spearman Correlation 1:\n", spearman_corr1)
    print("Spearman Correlation 2:\n", spearman_corr2)
    print("Spearman Correlation Difference:\n", spearman_diff)

    # Get dataset sizes
    size1 = df1.shape
    size2 = df2.shape

    # Create a subplot figure with Plotly
    num_histograms = len(display_columns1)
    fig = make_subplots(
        rows=4 + num_histograms, cols=2,
        subplot_titles=(f'Pearson Correlation (Severed Haltere - {size1[0]} rows, {size1[1]} columns)',
                        f'Pearson Correlation (Good Haltere - {size2[0]} rows, {size2[1]} columns)',
                        f'Spearman Correlation (Severed Haltere - {size1[0]} rows, {size1[1]} columns)',
                        f'Spearman Correlation (Good Haltere - {size2[0]} rows, {size2[1]} columns)',
                        'Pearson Correlation Difference',
                        'Spearman Correlation Difference',
                        'Statistics (Severed Haltere)',
                        'Statistics (Good Haltere)'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "table"}, {"type": "table"}]] + [
                  [{"type": "histogram"}, {"type": "histogram"}] for _ in range(num_histograms)],
        column_widths=[0.5, 0.5],
        row_heights=[0.2, 0.2, 0.2, 0.15] + [0.1] * num_histograms
    )

    # Pearson Correlation Heatmap for Severed Haltere
    fig.add_trace(
        go.Heatmap(
            z=pearson_corr1.values,
            x=display_columns1,
            y=display_columns1,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_corr1.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=1, col=1
    )

    # Pearson Correlation Heatmap for Good Haltere
    fig.add_trace(
        go.Heatmap(
            z=pearson_corr2.values,
            x=display_columns2,
            y=display_columns2,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_corr2.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=1, col=2
    )

    # Spearman Correlation Heatmap for Severed Haltere
    fig.add_trace(
        go.Heatmap(
            z=spearman_corr1.values,
            x=display_columns1,
            y=display_columns1,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_corr1.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=2, col=1
    )

    # Spearman Correlation Heatmap for Good Haltere
    fig.add_trace(
        go.Heatmap(
            z=spearman_corr2.values,
            x=display_columns2,
            y=display_columns2,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_corr2.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=2, col=2
    )

    # Pearson Correlation Difference
    fig.add_trace(
        go.Heatmap(
            z=pearson_diff.values,
            x=display_columns1,
            y=display_columns1,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_diff.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Difference: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=3, col=1
    )

    # Spearman Correlation Difference
    fig.add_trace(
        go.Heatmap(
            z=spearman_diff.values,
            x=display_columns1,
            y=display_columns1,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_diff.round(2).values,
            hoverinfo='text',
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Difference: %{z:.2f}<extra></extra>',
            showscale=True
        ),
        row=3, col=2
    )

    # Statistics table for Severed Haltere
    fig.add_trace(
        go.Table(
            header=dict(values=['Variable', 'Mean', 'Standard Deviation'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[stats1[col] for col in ['Variable', 'Mean', 'Standard Deviation']],
                       fill_color='lavender',
                       align='left')
        ),
        row=4, col=1
    )

    # Statistics table for Good Haltere
    fig.add_trace(
        go.Table(
            header=dict(values=['Variable', 'Mean', 'Standard Deviation'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[stats2[col] for col in ['Variable', 'Mean', 'Standard Deviation']],
                       fill_color='lavender',
                       align='left')
        ),
        row=4, col=2
    )

    # Histograms for Severed and Good Haltere
    for idx, col in enumerate(display_columns1):
        fig.add_trace(
            go.Histogram(
                x=df_numeric1[col],
                name=col,
                opacity=0.75,
                hoverinfo='text',
                hovertemplate=f'<b>{col} (Severed Haltere)</b><br>Value: {{x}}<br>Count: {{y}}<extra></extra>'
            ),
            row=5 + idx, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=df_numeric2[col],
                name=col,
                opacity=0.75,
                hoverinfo='text',
                hovertemplate=f'<b>{col} (Good Haltere)</b><br>Value: {{x}}<br>Count: {{y}}<extra></extra>'
            ),
            row=5 + idx, col=2
        )

        fig.update_xaxes(title_text=col, row=5 + idx, col=1)
        fig.update_xaxes(title_text=col, row=5 + idx, col=2)

    fig.update_layout(height=2500 + 200 * num_histograms, width=2000, title_text='Comparison of Experiments')
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickangle=0)

    fig.write_html(output_file)
    print(f'Comparison heatmaps, statistics tables, and histograms saved to {output_file}')


def compute_correlations_consecutive_wingbits(csv_path, save_name="correlations_consecutive_wingbits.html"):
    def confidence_interval(r, n, alpha=0.05):
        # Fisher Z-transformation
        Z = np.arctanh(r)

        # Standard error
        SE_Z = 1 / np.sqrt(n - 3)

        # Confidence interval in Z-space
        Z_critical = stats.norm.ppf(1 - alpha / 2)
        Z_lower = Z - Z_critical * SE_Z
        Z_upper = Z + Z_critical * SE_Z

        # Inverse Fisher Z-transformation
        r_lower = np.tanh(Z_lower)
        r_upper = np.tanh(Z_upper)

        return r_lower, r_upper

    print(f"Computes correlations for consecutive wingbits\n{csv_path}")
    df = pd.read_csv(csv_path)

    # Extract movie identifiers and wingbit numbers
    df['movie'] = df['wingbit'].str.extract(r'(dir\d+_mov\d+)')
    df['wingbit_num'] = df['wingbit'].str.extract(r'(\d+)$').astype(int)

    # Create empty dataframes to store combined correlations
    body_columns = [col for col in df.columns if col.startswith('body')]
    angle_columns = [col for col in df.columns if col.startswith(('theta', 'phi', 'psi'))]
    pearson_corr_combined = pd.DataFrame(0.0, index=body_columns, columns=angle_columns)
    spearman_corr_combined = pd.DataFrame(0.0, index=body_columns, columns=angle_columns)
    count = pd.DataFrame(0, index=body_columns, columns=angle_columns)

    # Process each movie separately and store individual statistics
    movie_stats = {}
    for movie in df['movie'].unique():
        df_movie = df[df['movie'] == movie].sort_values(by='wingbit_num')

        # Lists to accumulate pairs of values
        pearson_values = {body_col: {angle_col: [] for angle_col in angle_columns} for body_col in body_columns}
        spearman_values = {body_col: {angle_col: [] for angle_col in angle_columns} for body_col in body_columns}

        for i in range(len(df_movie) - 1):
            current_row = df_movie.iloc[i]
            next_row = df_movie.iloc[i + 1]

            for body_col in body_columns:
                for angle_col in angle_columns:
                    try:
                        current_val = float(current_row[body_col])
                        next_val = float(next_row[angle_col])
                    except ValueError:
                        continue
                    if not np.isnan(current_val) and not np.isnan(next_val):
                        pearson_values[body_col][angle_col].append((current_val, next_val))
                        spearman_values[body_col][angle_col].append((current_val, next_val))

        # Compute correlations for each combination of columns
        pearson_corr_movie = pd.DataFrame(index=body_columns, columns=angle_columns)
        spearman_corr_movie = pd.DataFrame(index=body_columns, columns=angle_columns)
        count_movie = pd.DataFrame(0, index=body_columns, columns=angle_columns)

        for body_col in body_columns:
            for angle_col in angle_columns:
                if pearson_values[body_col][angle_col]:
                    current_vals, next_vals = zip(*pearson_values[body_col][angle_col])
                    r_pearson = np.corrcoef(current_vals, next_vals)[0, 1]
                    r_spearman, _ = stats.spearmanr(current_vals, next_vals)

                    pearson_corr_movie.loc[body_col, angle_col] = r_pearson
                    spearman_corr_movie.loc[body_col, angle_col] = r_spearman
                    count_movie.loc[body_col, angle_col] = len(current_vals)

        # Combine the movie correlations
        pearson_corr_combined += pearson_corr_movie.fillna(0)
        spearman_corr_combined += spearman_corr_movie.fillna(0)
        count += count_movie

        # Store the statistics for the movie
        stats_df_movie = df_movie.describe().loc[['mean', 'std']].T.reset_index()
        stats_df_movie['mean'] = stats_df_movie['mean'].round(2)
        stats_df_movie['std'] = stats_df_movie['std'].round(2)
        stats_df_movie.columns = ['Variable', 'Mean', 'Standard Deviation']
        movie_stats[movie] = stats_df_movie

    # Average the combined correlations
    pearson_corr_combined /= count
    spearman_corr_combined /= count

    # Calculate mean and std for the numerical columns, rounded to 2 significant digits
    stats_df_combined = df.describe().loc[['mean', 'std']].T.reset_index()
    stats_df_combined['mean'] = stats_df_combined['mean'].round(2)
    stats_df_combined['std'] = stats_df_combined['std'].round(2)
    stats_df_combined.columns = ['Variable', 'Mean', 'Standard Deviation']

    # Compute confidence intervals for Pearson and Spearman correlations
    num_rows = count.max().max()  # Use the maximum count for confidence interval calculation
    pearson_confidence_intervals = np.zeros((len(body_columns), len(angle_columns), 2))
    spearman_confidence_intervals = np.zeros((len(body_columns), len(angle_columns), 2))

    for i, body_col in enumerate(body_columns):
        for j, angle_col in enumerate(angle_columns):
            r_pearson = pearson_corr_combined.loc[body_col, angle_col]
            r_spearman = spearman_corr_combined.loc[body_col, angle_col]
            pearson_confidence_intervals[i, j] = confidence_interval(r_pearson, num_rows)
            spearman_confidence_intervals[i, j] = confidence_interval(r_spearman, num_rows)

    # Create a subplot figure with Plotly
    fig = make_subplots(
        rows=len(movie_stats) + 3, cols=1,
        subplot_titles=(['Pearson Correlation', 'Spearman Correlation'] + [f'Statistics for {movie}' for movie in
                                                                           movie_stats.keys()] + [
                            'Combined Statistics']),
        specs=[[{"type": "heatmap"}], [{"type": "heatmap"}]] + [[{"type": "table"}]] * len(movie_stats) + [
            [{"type": "table"}]]
    )

    # Format hover text for Pearson correlation
    pearson_hover_text = [[
        (f"<b>{body_columns[i]}</b> vs <b>{angle_columns[j]}</b><br>Correlation: "
         f"{pearson_corr_combined.iloc[i, j]:.2f}<br>95% CI: [{pearson_confidence_intervals[i, j, 0]:.2f}, "
         f"{pearson_confidence_intervals[i, j, 1]:.2f}]")
        for j in range(len(angle_columns))] for i in range(len(body_columns))]

    # Pearson Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pearson_corr_combined.values,
            x=angle_columns,
            y=body_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=1, col=1
    )

    # Format hover text for Spearman correlation
    spearman_hover_text = [[
        f"<b>{body_columns[i]}</b> vs <b>{angle_columns[j]}</b><br>Correlation: {spearman_corr_combined.iloc[i, j]:.2f}<br>95% CI: [{spearman_confidence_intervals[i, j, 0]:.2f}, {spearman_confidence_intervals[i, j, 1]:.2f}]"
        for j in range(len(angle_columns))] for i in range(len(body_columns))]

    # Spearman Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=spearman_corr_combined.values,
            x=angle_columns,
            y=body_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=2, col=1
    )

    # Add statistics tables for each movie
    current_row = 3
    for movie, stats_df_movie in movie_stats.items():
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_df_movie.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[stats_df_movie[col] for col in stats_df_movie.columns],
                           fill_color='lavender',
                           align='left')
            ),
            row=current_row, col=1
        )
        current_row += 1

    # Adding the combined statistics table
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_df_combined.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[stats_df_combined[col] for col in stats_df_combined.columns],
                       fill_color='lavender',
                       align='left')
        ),
        row=current_row, col=1
    )

    fig.update_layout(height=1500 + 300 * len(movie_stats), width=1000,
                      title_text=f'Correlation Heatmaps and Statistics (Data size: {num_rows} rows, {len(df.columns)} columns)')
    save_path = os.path.join(os.path.dirname(csv_path), save_name)
    fig.write_html(save_path)

    print(f'Correlation heatmaps and statistics table saved to {save_path}')


def compute_shifted_correlations(file_path, save_name="correlations.html"):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # remove the word 'take' from the attributes names
    df.rename(columns=lambda x: x.replace('_take', ''), inplace=True)

    # Extract the wingbit number and movie identifier
    df['wingbit_num'] = df['wingbit'].str.extract(r'(\d+)$').astype(int)
    df['movie'] = df['wingbit'].str.extract(r'^(.*)_wingbit_\d+$')

    # Sort the dataframe by movie and wingbit_num to ensure proper shifting
    df = df.sort_values(by=['movie', 'wingbit_num'])

    # Shift the dataframe up by one row
    shifted_df = df.shift(-1)

    # Identify rows where the next row is from a different movie
    mask = (df['movie'] == shifted_df['movie'])

    # Apply the mask to filter out invalid correlations
    valid_df = df[mask]
    valid_shifted_df = shifted_df[mask]

    # Select columns of interest for correlation
    valid_columns = [col for col in valid_df.columns if col.startswith(('psi', 'phi', 'theta'))]
    shifted_columns = [col for col in valid_shifted_df.columns if col.startswith('body')]

    # Convert columns to numeric and fill NaN values
    valid_df[valid_columns] = valid_df[valid_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    valid_shifted_df[shifted_columns] = valid_shifted_df[shifted_columns].apply(pd.to_numeric, errors='coerce').fillna(
        0)

    # Initialize DataFrames to store the Pearson and Spearman correlation results
    pearson_correlation_matrix = pd.DataFrame(index=valid_columns, columns=shifted_columns)
    spearman_correlation_matrix = pd.DataFrame(index=valid_columns, columns=shifted_columns)

    # Initialize arrays for confidence intervals
    pearson_confidence_intervals = np.zeros((len(valid_columns), len(shifted_columns), 2))
    spearman_confidence_intervals = np.zeros((len(valid_columns), len(shifted_columns), 2))

    # Compute the Pearson and Spearman correlations for each attribute pair
    for i, column1 in enumerate(valid_columns):
        for j, column2 in enumerate(shifted_columns):
            col1 = valid_df[column1].values
            col2 = valid_shifted_df[column2].values

            pearson_corr, _ = pearsonr(col1, col2)
            spearman_corr, _ = spearmanr(col1, col2)

            pearson_correlation_matrix.loc[column1, column2] = pearson_corr
            spearman_correlation_matrix.loc[column1, column2] = spearman_corr

            pearson_confidence_intervals[i, j] = confidence_interval(pearson_corr, len(col1))
            spearman_confidence_intervals[i, j] = confidence_interval(spearman_corr, len(col1))

    # Create a subplot figure with Plotly
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Pearson Correlation', 'Spearman Correlation', 'Statistics'),
        specs=[[{"type": "heatmap"}], [{"type": "heatmap"}], [{"type": "table"}]]
    )

    # Format hover text for Pearson correlation
    pearson_hover_text = [[
        (f"<b>{valid_columns[i]}</b> vs <b>{shifted_columns[j]}</b><br>Correlation: "
         f"{pearson_correlation_matrix.iloc[i, j]:.2f}<br>95% CI: [{pearson_confidence_intervals[i, j, 0]:.2f}, "
         f"{pearson_confidence_intervals[i, j, 1]:.2f}]")
        for j in range(len(shifted_columns))] for i in range(len(valid_columns))]

    # Pearson Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pearson_correlation_matrix.values.astype(float),
            x=shifted_columns,
            y=valid_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=pearson_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=1, col=1
    )

    # Format hover text for Spearman correlation
    spearman_hover_text = [[
        (f"<b>{valid_columns[i]}</b> vs <b>{shifted_columns[j]}</b><br>Correlation: "
         f"{spearman_correlation_matrix.iloc[i, j]:.2f}<br>95% CI: [{spearman_confidence_intervals[i, j, 0]:.2f}, "
         f"{spearman_confidence_intervals[i, j, 1]:.2f}]")
        for j in range(len(shifted_columns))] for i in range(len(valid_columns))]

    # Spearman Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=spearman_correlation_matrix.values.astype(float),
            x=shifted_columns,
            y=valid_columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=spearman_hover_text,
            hoverinfo='text',
            showscale=True
        ),
        row=2, col=1
    )

    # Compute statistics
    stats_df = valid_df[valid_columns].describe().loc[['mean', 'std']].T.reset_index()
    stats_df['mean'] = stats_df['mean'].round(2)
    stats_df['std'] = stats_df['std'].round(2)
    stats_df.columns = ['Variable', 'Mean', 'Standard Deviation']

    # Adding the statistics table
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[stats_df[col] for col in stats_df.columns],
                       fill_color='lavender',
                       align='left')
        ),
        row=3, col=1
    )

    fig.update_layout(height=1500, width=1000,
                      title_text=save_name[:-5])

    # Save the results to new CSV files in the same directory as the input file
    output_dir = os.path.dirname(file_path)
    save_path = os.path.join(output_dir, save_name)
    fig.write_html(save_path)

    print(f'Correlation heatmaps and statistics table saved to {save_path}')



def add_nans(original_array, num_of_nans):
    # Create an array of NaNs with size M, 3
    nan_frames = np.full((num_of_nans, 3), np.nan)
    # Concatenate the NaN frames with the original array
    new_array = np.vstack((nan_frames, original_array))
    return new_array


def compare_autocorrelations_before_and_after_dark(input_hdf5_path, T=20):
    dir = os.path.dirname(input_hdf5_path)
    fig = go.Figure()

    with h5py.File(input_hdf5_path, 'r') as hdf:
        movies = list(hdf.keys())
        for movie in movies:
            print(movie)
            group = hdf[movie]
            x_body = group['x_body'][:]
            y_body = group['y_body'][:]
            z_body = group['z_body'][:]

            first_analysed_frame = int(group['first_analysed_frame'][()])
            first_y_body_frame = int(group['first_y_body_frame'][()])
            start_frame = first_y_body_frame + first_analysed_frame

            end_frame_second_part = int(group['end_frame'][()]) + first_analysed_frame
            end_frame_first_part = T * 16

            if start_frame + 100 >= end_frame_first_part or len(x_body) < 400:
                continue

            # first part
            x_body_first_part = x_body[start_frame:end_frame_first_part]
            y_body_first_part = y_body[start_frame:end_frame_first_part]
            z_body_first_part = z_body[start_frame:end_frame_first_part]
            AC_first = FlightAnalysis.get_auto_correlation_x_body(x_body_first_part)

            # second part
            x_body_second_part = x_body[end_frame_first_part:end_frame_second_part]
            y_body_second_part = y_body[end_frame_first_part:end_frame_second_part]
            z_body_second_part = z_body[end_frame_first_part:end_frame_second_part]
            AC_second = FlightAnalysis.get_auto_correlation_x_body(x_body_second_part)

            fig.add_trace(go.Scatter(x=list(np.arange(len(AC_first)) / 16), y=AC_first,
                                     mode='lines', name=f'{movie} First Part',
                                     line=dict(color='red')))
            fig.add_trace(go.Scatter(x=list(np.arange(len(AC_second)) / 16), y=AC_second,
                                     mode='lines', name=f'{movie} Second Part',
                                     line=dict(color='blue')))

    fig.update_layout(title='Autocorrelation Before and After Dark for Each Movie',
                      xaxis_title='ms',
                      yaxis_title='Autocorrelation',
                      legend_title='Movies')
    html_out_path = os.path.join(dir, 'autocorrelation_plot.html')
    fig.write_html(html_out_path)


def create_correlations_cluster():
    # base_path = "roni dark 60ms"
    base_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms"
    save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)
    output_file = os.path.join(base_path, "combined_wingbits.h5")
    combine_wingbits_and_save(output_file, base_path)
    csv_file = os.path.join(base_path, "all_wingbits_attributes_good_haltere.csv")
    create_dataframe_from_h5(output_file, "all_wingbits_attributes_good_haltere.csv")
    csv_path_good_haltere = os.path.join(base_path, "all_wingbits_attributes_good_haltere.csv")
    compute_correlations(csv_path_good_haltere, "corretaions_good_Haltere.html")
    create_histograms(csv_path_good_haltere, "Histograms_good_Haltere.html")
    compute_shifted_correlations(csv_path_good_haltere, "correlations between consecutive wingbits good haltere.html")


def create_correlations_from_drive(only_correlations=False):
    if not only_correlations:
        base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
        save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)
        base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
        save_movies_data_to_hdf5(base_path, output_hdf5_path="", smooth=True, one_h5_for_all=False)
        dir1 = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
        dir2 = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\dark 24-1 movies"
        output_file = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\combined_wingbits.h5"
        # Combine and save the wingbits
        combine_wingbits_and_save(output_file, dir1, dir2)
        create_dataframe_from_h5(output_file, "all_wingbits_attributes_severed_haltere.csv")
    base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data"
    csv_path_severed_haltere = os.path.join(base_path, "all_wingbits_attributes_severed_haltere.csv")
    compute_shifted_correlations(csv_path_severed_haltere, "correlations between consecutive wingbits bad haltere.html")
    compute_correlations(csv_path_severed_haltere, "corretaions_severed_Haltere.html")
    create_histograms(csv_path_severed_haltere, "Histograms_severed_Haltere.html")