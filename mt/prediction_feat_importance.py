import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mt.definitions import DATA_DIR

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "final.csv", index_col=0)
    # Assuming df is your DataFrame and 'Y_scaled' is the target
    X = df.drop(
        columns=["Y", "Y_scale", "Y_scaled", "commit_index"]
    )  # or any non-feature columns
    y = df["Y_scaled"]  # or 'Y' or 'Y_standard_scaled' based on your target definition

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the linear regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_df = pd.DataFrame({"Mean Squared Error": [mse], "R^2 Score": [r2]})
    metrics_df.to_csv("metrics.csv")

    # Feature Importances
    rf_importances = model.feature_importances_
    importances_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf_importances}
    ).sort_values(by="Importance", ascending=False)

    importances_df.to_csv("importances.csv")
    # # Extracting feature importances (coefficients)
    # feature_importances = model.coef_

    # # Create a DataFrame for feature importances
    # importances_df = pd.DataFrame(
    #     {"Feature": X.columns, "Importance": feature_importances}
    # )
    # # Sort by importance
    # importances_df.sort_values(by="Importance", ascending=False, inplace=True)

    # importances_df.to_csv("importances.csv")

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--"
    )  # Diagonal line
    plt.savefig("pred.png")
