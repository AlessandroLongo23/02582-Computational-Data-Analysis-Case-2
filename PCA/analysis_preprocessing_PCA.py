import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def summary_info(df):
    # Basic Info
    print("\nBasic Information:")
    print(df.info())

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

    # Check data types
    print("\nData Types:")
    print(df.dtypes)



def fill_missing_values(df) -> pd.DataFrame:
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Show rows with missing values
    print("\nRows with Missing Values:")
    print(df[df.isnull().any(axis=1)])

    # Show rows with missing values and the columns where they're missing
    print("\nRows with Missing Values and Missing Columns:")
    rows_with_na = df[df.isnull().any(axis=1)]
    for index, row in rows_with_na.iterrows():
        missing_cols = row[row.isnull()].index.tolist()
        print(f"Row {index} is missing values in columns: {missing_cols}")

    # Fill missing values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
            print(f"Filled missing numeric values in '{column}' with mean: {mean_value}")
        else:
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else "N/A"
            df[column].fillna(mode_value, inplace=True)
            print(f"Filled missing non-numeric values in '{column}' with mode: {mode_value}")

    print("\nAll missing values have been filled.")
    return df


def drop_missing_values(df) -> pd.DataFrame:
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Show rows with missing values
    print("\nRows with Missing Values:")
    print(df[df.isnull().any(axis=1)])

    # Show rows with missing values and the columns where they're missing
    print("\nRows with Missing Values and Missing Columns:")
    rows_with_na = df[df.isnull().any(axis=1)]
    for index, row in rows_with_na.iterrows():
        missing_cols = row[row.isnull()].index.tolist()
        print(f"Row {index} is missing values in columns: {missing_cols}")

    # Drop rows with missing values
    df = df.dropna()
    print("\nRows with missing values have been dropped.")
    print(f"Remaining rows: {df.shape[0]}")

    return df


def visualize_data(df):
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    num_cols = len(corr_matrix.columns)
    num_rows = len(corr_matrix.index)
    plt.xticks(np.arange(0.5, num_cols, 1), corr_matrix.columns, rotation=90, ha='center', fontsize=5)
    plt.yticks(np.arange(0.5, num_rows, 1), corr_matrix.index, rotation=0, va='center', fontsize=5)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

    # Histograms for numerical features
    print("\nGenerating Histograms...")
    # Create histogram subplots
    axes = df.hist(bins=20, figsize=(15, 10))
    
    # Adjust title and tick label font sizes for each subplot
    for ax_row in axes:
        for ax in ax_row:
            ax.set_title(ax.get_title(), fontsize=10)  # Set font size for subplot titles
            ax.tick_params(axis='both', which='major', labelsize=8)  # Set font size for tick labels
            
            # Update font size for x and y axis numbers
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
    
    plt.suptitle("Histograms of Numerical Features", fontsize=12)
    plt.tight_layout(pad=1.8)
    plt.show()

    # Count plots for categorical features
    # print("\n📋 Generating Count Plots for Categorical Features...")
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # for col in categorical_cols:
    #     plt.figure(figsize=(8, 4))
    #     sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    #     plt.title(f"Count Plot of {col}")
    #     plt.xticks(rotation=45)
    #     plt.show()


def perform_pca(df):
    print("\nPerforming PCA for Dimensionality Reduction...")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # Select only numerical columns for PCA
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    
    # Get the maximum number of components (equal to number of features)
    n_components = min(numerical_df.shape)
    
    # Create PCA instance with maximum components
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Print variance explained by each component
    print("\nVariance Explained by Each Principal Component:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance', zorder=3)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual Explained Variance', zorder=2)
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% Explained Variance', zorder=2)
    plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Explained Variance', zorder=2)
    plt.axhline(y=0.90, color='gold', linestyle='--', label='90% Explained Variance', zorder=2)
    plt.axhline(y=0.80, color='g', linestyle='--', label='80% Explained Variance', zorder=2)
    plt.axhline(y=0.70, color='b', linestyle='--', label='70% Explained Variance', zorder=2)
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Find number of components needed for different variance thresholds
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
    components_needed = {}
    
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_needed[threshold] = n_components
        print(f"Number of components needed for {threshold*100:.0f}% variance: {n_components}")
    
    # Plot first two principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('First Two Principal Components')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Return the PCA results for potential further analysis
    return pca, pca_data, numerical_df.columns


def main():
    df = load_data("HR_data.csv")

    summary_info(df)

    df = df.drop(columns=[df.columns[0]])   # Drop the first (ID) column
    df = handle_missing_values(df)

    # Save cleaned dataset
    df.to_csv('HR_data_cleaned.csv', index=False)

    visualize_data(df)

    df = load_data("HR_data_cleaned.csv")

    perform_pca(df)


if __name__ == "__main__":
    main()
