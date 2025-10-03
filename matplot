import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    try:
        # =======================
        # Task 1: Load & Explore
        # =======================
        print("Loading dataset...")

        # Load Iris dataset from sklearn
        iris = load_iris(as_frame=True)
        df = iris.frame  # Pandas DataFrame with features + target
        df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

        # Display first few rows
        print("\nFirst 5 rows of dataset:")
        print(df.head())

        # Check structure
        print("\nDataset Info:")
        print(df.info())

        print("\nMissing values in each column:")
        print(df.isnull().sum())

        # Clean dataset (if missing values exist)
        df = df.dropna()
        print("\nAfter cleaning, dataset shape:", df.shape)

        # =======================
        # Task 2: Basic Analysis
        # =======================
        print("\nBasic statistics:")
        print(df.describe())

        # Group by species and compute mean
        print("\nMean values grouped by species:")
        print(df.groupby("species").mean())

        # Example finding
        print("\nObservation: Setosa flowers generally have smaller petal sizes than Versicolor and Virginica.")

        # =======================
        # Task 3: Visualizations
        # =======================

        # Line chart (example: cumulative petal length to simulate trend)
        plt.figure(figsize=(8,5))
        df['petal length (cm)'].cumsum().plot(kind='line')
        plt.title("Cumulative Petal Length Over Samples")
        plt.xlabel("Sample Index")
        plt.ylabel("Cumulative Petal Length (cm)")
        plt.grid(True)
        plt.show()

        # Bar chart (average petal length per species)
        plt.figure(figsize=(8,5))
        sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="viridis")
        plt.title("Average Petal Length by Species")
        plt.xlabel("Species")
        plt.ylabel("Average Petal Length (cm)")
        plt.show()

        # Histogram (distribution of sepal width)
        plt.figure(figsize=(8,5))
        plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
        plt.title("Distribution of Sepal Width")
        plt.xlabel("Sepal Width (cm)")
        plt.ylabel("Frequency")
        plt.show()

        # Scatter plot (sepal length vs petal length)
        plt.figure(figsize=(8,5))
        sns.scatterplot(
            x="sepal length (cm)", 
            y="petal length (cm)", 
            hue="species", 
            style="species", 
            palette="deep", 
            data=df
        )
        plt.title("Sepal Length vs Petal Length by Species")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Petal Length (cm)")
        plt.legend(title="Species")
        plt.show()

    except FileNotFoundError:
        print("❌ Error: Dataset file not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
