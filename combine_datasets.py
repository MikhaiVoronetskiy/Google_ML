import pandas as pd

def extract_year(date):
    year_list = []
    for i in date:
        if i.isdigit():
            year_list.append(i)
            if len(year_list) == 4:
                return int("".join(map(str, year_list)))
        else:
            year_list = []
    return None

def extract_runtime(runtime):
    runtime_list = []
    for i in runtime:
        if i.isdigit():
            runtime_list.append(i)
        else:
            return int("".join(map(str, runtime_list)))
    return None

def extract_genre(genre):
    genre_list = []
    for i in genre:
        if i.isalpha():
            genre_list.append(i)
        else:
            return "".join(map(str, genre_list))
    return "".join(map(str, genre_list))

def extract_name(director):
    director_list = []
    for i in director:
        if i.isalpha() or i == " ":
            director_list.append(i)
        elif len(director_list) > 0:
            return "".join(map(str, director_list))
    return "".join(map(str, director_list))

def create_combined3_dataset():
    # Load the initial dataset
    initial_dataset_path = 'InitialDataset.csv'
    initial_df = pd.read_csv(initial_dataset_path)

    # Load the additional dataset
    five_thousand_dataset_path = '5000_Count.csv'
    five_thousand = pd.read_csv(five_thousand_dataset_path)

    # Load the third dataset
    thousand_dataset_path = '10000_Director_stars_duration_ratings .csv'
    thousand_df = pd.read_csv(thousand_dataset_path)

    # Apply the extract_year function to the 'Release Date' column
    initial_df['Release Date'] = initial_df['Release Date'].apply(lambda x: extract_year(x) if pd.notna(x) else x)
    five_thousand['Release Date'] = five_thousand['Release Date'].apply(lambda x: extract_year(x) if pd.notna(x) else x)

    # Ensure 'IMDB Rating' columns are of the same type
    five_thousand['IMDB Rating'] = five_thousand['IMDB Rating'].astype(float)
    thousand_df['IMDB Rating'] = thousand_df['IMDB Rating'].astype(float)

    # Merge the datasets based on a common key (e.g., movie title or ID)
    merged_df = thousand_df.merge(five_thousand, on=['Movie', 'IMDB Rating'], how='left')
    merged_df = initial_df.merge(merged_df, on=['Movie', 'Release Date'], how='left')

    merged_df = merged_df.dropna()
    merged_df['Runtime'] = merged_df['Runtime'].apply(lambda x: extract_runtime(x) if pd.notna(x) else x)
    #merged_df['Release Date'] = merged_df['Release Date'].apply(lambda x: extract_year(x) if pd.notna(x) else x)
    merged_df['Genre'] = merged_df['Genre'].apply(lambda x: extract_genre(x))
    merged_df['Directors'] = merged_df['Directors'].apply(lambda x: extract_name(x))
    merged_df['Country of Origin'] = merged_df['Country of Origin'].apply(lambda x: extract_name(x))

    # Save the new dataset to a file
    new_dataset_path = 'new_dataset.csv'
    merged_df.to_csv(new_dataset_path, index=False)


def merge_initial_with_10000():
    initial_dataset_path = 'InitialDataset.csv'
    initial_df = pd.read_csv(initial_dataset_path)

    thousand_dataset_path = '10000_Director_stars_duration_ratings .csv'
    thousand_df = pd.read_csv(thousand_dataset_path)

    merged_df = initial_df.merge(thousand_df, on=['Movie'], how='left')
    merged_df = merged_df.dropna()
    new_dataset_path = '2new_dataset.csv'

    merged_df['Runtime'] = merged_df['Runtime'].apply(lambda x: extract_runtime(x))
    merged_df['Release Date'] = merged_df['Release Date'].apply(lambda x: extract_year(x) if pd.notna(x) else x)
    merged_df['Genre'] = merged_df['Genre'].apply(lambda x: extract_genre(x))
    merged_df['Directors'] = merged_df['Directors'].apply(lambda x: extract_name(x))


    merged_df = merged_df.drop_duplicates(subset=['Movie'], keep=False)
    merged_df.to_csv(new_dataset_path, index=False)

create_combined3_dataset()