import pandas as pd
import numpy as np
from datetime import datetime

def prepare_data(df):
    # Step 1 - Order the Data

    ### Duplicates and Empty
    df = df.drop_duplicates()

    # Remove columns with many missing values
    columns_to_drop = ['Supply_score', 'Pic_num', 'Color', 'Area', 'City', 'Test', 'Description']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Convert date columns to datetime format if not already
    if 'Repub_date' in df.columns and df['Repub_date'].dtype == 'object':
        df['Repub_date'] = pd.to_datetime(df['Repub_date'], format='%d/%m/%Y', errors='coerce')
    if 'Cre_date' in df.columns and df['Cre_date'].dtype == 'object':
        df['Cre_date'] = pd.to_datetime(df['Cre_date'], format='%d/%m/%Y', errors='coerce')

    # Calculate days since for the date columns
    current_date = datetime.now()
    if 'Repub_date' in df.columns:
        df['Repub_date_days_since'] = (current_date - df['Repub_date']).dt.days
    if 'Cre_date' in df.columns:
        df['Cre_date_days_since'] = (current_date - df['Cre_date']).dt.days

    # Fill missing values for 'Repub_date_days_since' and 'Cre_date_days_since' using the mean value within groups
    if 'Repub_date_days_since' in df.columns:
        df['Repub_date_days_since'] = df.groupby(['manufactor', 'model', 'Year'])['Repub_date_days_since'].transform(lambda x: x.fillna(x.mean() if x.notna().sum() > 0 else df['Repub_date_days_since'].mean()))
    if 'Cre_date_days_since' in df.columns:
        df['Cre_date_days_since'] = df.groupby(['manufactor', 'model', 'Year'])['Cre_date_days_since'].transform(lambda x: x.fillna(x.mean() if x.notna().sum() > 0 else df['Cre_date_days_since'].mean()))

    # Drop the original date columns as they are no longer needed
    df = df.drop(columns=[col for col in ['Repub_date', 'Cre_date'] if col in df.columns])

    ### Manufactor Column
    if 'manufactor' in df.columns:
        df['manufactor'] = df['manufactor'].str.replace('Lexsus', 'לקסוס', regex=True)

    ### Model Column
    if 'model' in df.columns:
        for index, row in df.iterrows():
            manufactor = str(row['manufactor'])
            model = str(row['model'])
            if manufactor in model:
                df.at[index, 'model'] = model.replace(manufactor, '').strip()
        df['model'] = df['model'].str.strip()
        df['model'] = df['model'].str.extract(r'(\w+\s\w+|\w+\s\w+\s\w+|\w+)')
        df['model'] = df['model'].str.replace('CIVIC', 'סיוויק', regex=True)
        df['model'] = df['model'].str.replace('JAZZ', 'ג`אז', regex=True)
        df['model'] = df['model'].str.replace('ACCORD', 'אקורד', regex=True)
        df.loc[df['manufactor'] == 'הונדה', 'model'] = df.loc[df['manufactor'] == 'הונדה', 'model'].str.replace('INSIGHT', 'אינסייט', regex=True)
        df.loc[(df['manufactor'] == 'הונדה') & (df['model'] == 'האצ`בק'), 'model'] = 'סיוויק האצ`בק'
        df.loc[df['manufactor'] == 'הונדה', 'model'] = df.loc[df['manufactor'] == 'הונדה', 'model'].str.replace("האצ'בק", "האצ`בק", regex=True)
        df.loc[df['manufactor'] == 'הונדה', 'model'] = df.loc[df['manufactor'] == 'הונדה', 'model'].str.replace("ג'אז", "ג`אז", regex=True)
        df['model'] = df['model'].str.replace('אונסיס', 'אוונסיס', regex=True)
        df['model'] = df['model'].str.replace('קאונטרימן', 'קאנטרימן', regex=True)
        df['model'] = df['model'].str.replace('one', 'ONE', regex=True)
        df['model'] = df['model'].str.replace('מיטו / MITO', 'מיטו', regex=True)
        df['model'] = df['model'].str.replace('Taxi', '', regex=True)

    ### Gear Column
    if 'Gear' in df.columns:
        df['Gear'] = df['Gear'].replace('אוטומטי', 'אוטומטית')
        df['Gear'] = df.groupby(['manufactor', 'model'], group_keys=False)['Gear'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'אוטומטי'))

    ### Engine_type Column
    if 'Engine_type' in df.columns:
        df['Engine_type'] = df['Engine_type'].replace({'היבריד': 'היברידי', 'טורבו דיזל': 'דיזל'})
        df['Engine_type'] = df.groupby(['manufactor', 'model'], group_keys=False)['Engine_type'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'לא ידוע'))

    ### Prev_ownership and Curr_ownership
    if 'Prev_ownership' in df.columns:
        df['Prev_ownership'] = df.groupby(['manufactor', 'model'], group_keys=False)['Prev_ownership'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'לא ידוע'))
        df['Prev_ownership'] = df['Prev_ownership'].replace(['None', 'לא מוגדר'], 'לא ידוע')
    if 'Curr_ownership' in df.columns:
        df['Curr_ownership'] = df.groupby(['manufactor', 'model'], group_keys=False)['Curr_ownership'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'לא ידוע'))
        df['Curr_ownership'] = df['Curr_ownership'].replace(['None', 'לא מוגדר'], 'לא ידוע')

    ### Create Km_per_Year Column and Remove Km, Year Columns
    if 'Km' in df.columns:
        df['Km'] = df['Km'].astype(str).str.replace(',', '').apply(lambda x: float(x) if x.replace('.', '', 1).isdigit() else np.nan)
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Km' in df.columns and 'Year' in df.columns:
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
        current_year = datetime.now().year
        df['Car_Age'] = current_year - df['Year']
        df['Km_per_Year'] = df['Km'] / df['Car_Age']
        df['Km_per_Year'] = df['Km_per_Year'].round(1)
        df['Km_per_Year'] = df.groupby('Car_Age', group_keys=False)['Km_per_Year'].apply(lambda x: x.fillna(x.mean()))
        df = df.drop(columns=['Km', 'Year'])

    ### capacity_Engine Column
    if 'capacity_Engine' in df.columns:
        df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
        df.loc[df['capacity_Engine'] < 1000, 'capacity_Engine'] = df.loc[df['capacity_Engine'] < 1000, 'capacity_Engine'] * 10
        df['capacity_Engine'] = df.groupby(['manufactor', 'model'], group_keys=False)['capacity_Engine'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df['capacity_Engine'].mode()[0]))
        df['capacity_Engine'] = df['capacity_Engine'].astype(int)

    # encoding
    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Fill missing values after encoding
    df = df.fillna(0)

    # Rearrange columns so 'Price' is the last column
    if 'Price' in df.columns:
        price_column = df.pop('Price')
        df['Price'] = price_column

    return df

def get_columns():
    df = pd.read_csv('dataset.csv')
    columns = df.columns.tolist()
    if 'Supply_score' in columns:
        columns.remove('Supply_score')
    return columns

def get_unique_values():
    df = pd.read_csv('dataset.csv')
    df_prepared = prepare_data(df)

    unique_values = {}
    categorical_columns = ['model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership']

    for col in categorical_columns:
        if col in df_prepared.columns:
            unique_values[col] = df_prepared[col].unique().tolist()

    return unique_values

