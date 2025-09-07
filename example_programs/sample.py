import pandas as pd

class DataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe

    def clean_data(self):
        # Drop rows with missing values
        self.df.dropna(inplace=True)
        # Convert a column to numeric, coercing errors
        self.df['some_column'] = pd.to_numeric(self.df['some_column'], errors='coerce')
        return self.df

    def analyze_data(self):
        # Calculate descriptive statistics
        description = self.df.describe()
        # Get the shape of the dataframe
        shape = self.df.shape
        return description, shape

# Example usage
data = {'col1': [1, 2, None, 4], 'some_column': ['10', '20', '30', 'forty']}
df = pd.DataFrame(data)

processor = DataProcessor(df)
cleaned_df = processor.clean_data()
description, shape = processor.analyze_data()

print(cleaned_df)
print(description)
print(shape)