import random
from typing import Union
import numpy as np
import seaborn as sns
import pandas as pd
import re
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress


class Mice_Inspection():
    """
    A class for inspecting and formatting data related to mice species observations.
    
    Attributes:
    -----------
    op : list
        Output paths for storing processed data.
    subjects : int
        Number of subjects (mice) to process.
    sample_size : int
        Sample size for the observations.
    tot_species : int
        Total number of unique species found in the dataset.
    species_list : list
        List of unique species.
    os_dict : dict
        Mapping of OTUs (Operational Taxonomic Units) to species names.
    mice_df : list of pd.DataFrame
        Dataframes containing the mice species counts data.
    metadata_df : pd.DataFrame
        Dataframe containing metadata on OTUs and species.

    Methods:
    --------
    get_metadata(mdp, verbose=False):
        Loads and processes metadata on species, initializes related attributes.
        
    get_mice_df(sort_by='median'):
        Loads and processes the mice dataframes, sorting by 'median' or 'mean' counts.
    
    format_mice_dataframes(ip, op):
        Formats raw mice data and saves the results to specified output paths.
    
    transpose_mice_df():
        Transposes each mouse's dataframe, sorting species by abundance and filling missing days.
    
    interpolate_mice_df():
        Interpolates each mouse's dataframe with cubic splines method

    select_species():
        Returns the list of the species which appear with rank <= chosen rank in some mouse

    make_stacked_bar_plot():
        Makes stacked bar plot

    get_species_df(species):
        Creates and saves the aggregated data of species in a csv file where the first column is days and subsequent
        columns correspond to mice.
    
    plot_species(list):
        Plots the time series of abundances for the selected list of species. Returns pdf file(s) containing the plots.
        
    get_dissimilarities_df(mouse, write = True): 
        Computes and saves (only if write = True) dissimilarities for each species (rows) and for each time lag (columns) in the correspondent mouse dataset.
        The  output files are saved in the 'Data/dissimilarities' directory with filenames 'dissimilarity_<mouse>.csv'
        Only valid time lags (with computed dissimilarities) will appear as columns.
    
    def plot_dissimilarities_in_pdf(mice_diss, output_dir="Inspection_Outputs", n_species_per_plot=5, window_size=10, ma = True): 
        Plot all the dissimilarities on different pdfs (5 species at a time) for every mouse. 

    def params_to_pdf(mi, mice_ab, objective, bounds, LM, find_transient_end, file_path="parameters_data.npz", force_recompute=False)
        Save in pdf the parameters obtained with LM fit for all mice and all species
        
    def params_to_csv(mi, mice_ab, objective, bounds, LM, find_transient_end, force_recompute=False):
        This function processes the species abundance data for multiple mice, fits the logistic model 
        to each species' time series using optimization, computes theoretical estimates for the carrying 
        capacity (K), tau (time constant), and sigma (variance-related parameter), and saves the results 
        in CSV files.

    format_for_regression():

    Outputs: 
    --------
    Data:
        Directory containing formatted data.
    Inspection_Outputs/Stacked:
        Directory containing stacked bar plots.
    Inspection_Outputs/Species:
        Directory containing species-specific plots (es: time series).
    """

    def __init__(self, ip, op, mdp):
        """
        Initialize the MiceInspection class. Format data if output path does not exist,
        then load metadata and mice dataframes.

        Parameters:
        -----------
        input_paths : list
            List of input file paths for raw data.
        output_paths : list
            List of output file paths for storing processed data.
        metadata_path : str
            Path to the metadata CSV file.
        """
        self.output_dir = "Inspection_Outputs"
        os.makedirs(self.output_dir, exist_ok = True)
        self.op = op
        self.subjects = 8
        self.sample_size = 3000
        self.tot_species = 0
        self.species_list = None
        self.os_dict = None
        self.mice_df = None
        self.metadata_df = None
        self.get_metadata(mdp)
        if not os.path.exists(op[0]):
            os.makedirs("Data/by_mouse", exist_ok = True)
            self.format_mice_dataframes(ip, op)
        self.get_mice_df() # This method loads the processed data into self.mice_df

    def get_metadata(self, mdp, verbose = False):
        """
        Loads metadata, extracts OTU to species mapping, and initializes species-related attributes.
        
        Parameters:
        -----------
        metadata_path : str
            Path to the metadata CSV file.
        verbose : bool, optional
            If True, prints additional information about the species and OTUs (default is False).

        Returns:
        --------
        tuple : (os_dict, metadata_df, tot_species, species_list)
            os_dict : dict
                Mapping of OTUs to species.
            metadata_df : pd.DataFrame
                Metadata dataframe with columns ['query', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'].
            tot_species : int
                Total number of unique species.
            species_list : list
                List of unique species names.
        """
        metadata_df = pd.read_csv(mdp)
        self.metadata_df = metadata_df[["query", "Phylum", "Class", "Order", "Family", "Genus", "Species"]]
        otus_to_species_dict = {}
        for index, row in self.metadata_df.iterrows():
            otus_to_species_dict[row['query']] = row['Species']
        self.os_dict = otus_to_species_dict
        self.species_list = list(set(otus_to_species_dict.values()))
        # Fix species names
        species_to_fix = ['Clostridium sp. M62/1', 'Clostridium sp. SS2/1', 'Cryobacterium aff. psychrophilum A1/C-aer/OI',
                          'Hymenobacter sp. I/74-Cor2', 'Pseudoalteromonas sp. RE10F/2', 'Carnobacterium sp. 12266/2009']
        # Find the index corresponding of species 'Clostridium sp. M62/1' inside species_list
        for species in species_to_fix:
            species_index = self.species_list.index(species)
            species = species.replace("/", "_")
            self.species_list[species_index] = species
        self.tot_species = len(list(set(otus_to_species_dict.values())))
        if verbose:
            print(f"number of species: {len(list(set(otus_to_species_dict.values())))}")
            print(f"number of otus: {len(list(set(otus_to_species_dict.keys())))}")
        return self.os_dict, self.metadata_df, self.tot_species, self.species_list

    def get_mice_df(self, sort_by = 'median'):
        """
        Loads and processes the mice dataframes, sorting by 'median' or 'mean' counts.

        Parameters:
        -----------
        sort_by : str, optional
            Criteria for sorting the data ('median' or 'mean'). Default is 'median'.
        
        Returns:
        --------
        list of pd.DataFrame:
            List of dataframes, each corresponding to a subject's data.
        """
        if not sort_by in ['median', 'mean']:
            raise ValueError("sort_by must be 'median' or 'mean'")
        mice_df = []
        for n in range(self.subjects):
            df = pd.read_csv(self.op[n])
            # recast the column names containing days in integers for easier later access
            df.rename(columns= {col: int(col) for col in df.columns[1:]}, inplace=True)
            # Sort values
            if sort_by == 'mean':
                temp = df.iloc[:, 1:]
                df_sorted = df.loc[temp.mean(axis=1).sort_values(ascending = False).index]
                df_sorted.reset_index(drop = True, inplace = True)
            else:
                temp = df.iloc[:, 1:]
                df_sorted = df.loc[temp.median(axis=1).sort_values(ascending = False).index]
                df_sorted.reset_index(drop = True, inplace = True)
            mice_df.append(df_sorted)
        self.mice_df = mice_df
        return self.mice_df

    def format_mice_dataframes(self, ip, op):
        """
        Formats raw data from input paths and saves the processed data to output paths.

        Parameters:
        -----------
        input_paths : list
            List of input file paths containing raw data.
        output_paths : list
            List of output file paths for storing the formatted data.
        """
        for i in range(self.subjects):
            df = pd.read_csv(ip[i])
            # --------------------------------------------------- #
            # Rename columns and delete duplicate columns ------- #
            # --------------------------------------------------- #
            #there are some multiple measures taken on the same day
            #they correspond to column named as "MouseAllLife-C-XXXX-Y" (mouse 1)
            #or "MouseAllLife-C-XXX-Y" (others)
            #i discard measures except from the first one
            #(i could also compute the mean, but we'll see),
            #Define the dominant pattern to match 'MouseAllLife-C-XXXX' for mouse 1
            #and 'MouseAllLife-C-XXX' for the other mice
            if i==0:
                pattern = re.compile(r'^MouseAllLife-\d-\d{4}$')
                multiple_pattern = re.compile(r'^MouseAllLife-\d-\d{4}-\d$')
            else:
                pattern = re.compile(r'^MouseAllLife-\d-\d{3}$')
                multiple_pattern = re.compile(r'^MouseAllLife-\d-\d{3}-\d$')
            multiple_list = []
            for col in df.columns[1:]:
                if pattern.match(col):
                    parts = col.split('-')
                    df.rename(columns= {col: parts[-1]}, inplace=True)
                elif multiple_pattern.match(col):
                    parts = col.split('-')
                    time = parts[-2]
                    if time in multiple_list:
                        df.drop(columns= col, inplace=True)
                    else:
                        multiple_list.append(time)
                        df.rename(columns= {col: int(parts[-2])}, inplace=True)
            # --------------------------------------------------- #            
            # group otus corresponding to same species
            # --------------------------------------------------- # 
            for target in list(set(self.os_dict.values())):
                otus_list = [otus for otus, species in self.os_dict.items() if species == target]
                temp = df[df['otu'].isin(otus_list)]
                if not temp.empty:
                    sum_row = temp.iloc[:, 1:].sum(axis=0)  # Sums all columns except the first one
                    new_row = pd.Series([target] + sum_row.tolist(), index=df.columns)  # Create a new row
                    df.iloc[temp.index[0], :] = new_row  # Replace the first row
                    df.drop(index=temp.index[1:], inplace= True)
                    df.reset_index(drop=True, inplace=True)
            df.rename(columns={'otu': 'species'}, inplace = True)
            # -------------------------------------------------- #
            # Sort rows by mean counts 
            # --------------------------------------------------
            temp = df.iloc[:, 1:]
            df_sorted = df.loc[temp.sum(axis=1).sort_values().index].copy()
            df_sorted.reset_index(drop = True, inplace = True)
            # Save output
            df_sorted.to_csv(op[i], index = False)
        return

    def transpose_mice_df(self) -> None:
        """
        Transposes the mice dataframes and fills missing days with NaN values.
        Creates CSV outputs with day as the first column and species in subsequent columns.
        """
        for n in range(self.subjects):
            df =  self.mice_df[n].copy()
            days = df.columns[1:].to_list() 
            #df.drop(columns=['median_counts', 'mean_counts', 'Unnamed: 0'], inplace = True)
            df = df.set_index('species').transpose()
            df.reset_index(inplace = True, drop = True)
            df.insert(0, 'day', days)
            full_day_range = pd.Series(range(df['day'].min(), df['day'].max() + 1))
            df.set_index('day', inplace=True)
            df = df.reindex(full_day_range)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'day'}, inplace=True)
            if not os.path.exists('Data/by_mouse_transposed'):
                os.makedirs('Data/by_mouse_transposed')
            output_path = os.path.join("Data/by_mouse_transposed", f"mouse_{n + 1}_transposed.csv")
            df.to_csv(output_path, index = False)
        return 
    
    def sort_species(self, max_rank: int = None, sorting_criterion: str = 'mean', write_csv: bool = True) -> list:
        """
        For each species in the dataset, extract the mean (or median) count across all subjects. Computes the global mean (or median).
        Returns a .csv file with columns ['species', 'mean_subject_1', ... mean_subject_N', 'global_mean'] (or median).
        Returns also a list of the first max_rank species. If max_rank is None or outside the range, all species are returned.
        """
        if write_csv:
            output_dir = os.path.join('Data', 'by_species')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join('Data', 'by_species', f'{sorting_criterion}:{max_rank}_sorted.csv')
        if not max_rank is None:
            if not (isinstance(max_rank, int) and max_rank > 1):
                raise ValueError("max rank must be a positive integer")
        if not sorting_criterion in ['mean', 'median']:
            raise ValueError("sorting criterion must be 'mean' or 'median'")
        
        species_data = []
        for species in self.species_list:
            mean_counts = []
            median_counts = []
            for mouse in range(self.subjects):
                # If the species appear, extract the mean and median count
                temp = self.mice_df[mouse][self.mice_df[mouse]['species'] == species]
                if not temp.empty:
                    mean_counts.append(temp['mean_counts'].iloc[0])
                    median_counts.append(temp['median_counts'].iloc[0])
                else:
                    mean_counts.append(0)
                    median_counts.append(0)
            global_mean = np.mean(mean_counts)
            global_median = np.mean(median_counts)
            species_data.append({'species': species, 'global_mean': global_mean, 'global_median': global_median})
            df = pd.DataFrame(species_data)

        if sorting_criterion == 'median':
            df.sort_values(by= 'global_median', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
        if sorting_criterion == 'mean':
            df.sort_values(by= 'global_mean', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
                
        selected_species = df['species'].tolist()
        if max_rank is not None:
            max_rank = min(max_rank, len(df))
            # Cut the dataframe to the first max_rank species
            df = df.iloc[:max_rank]
            selected_species = df['species'].tolist()
        if write_csv:
            df.to_csv(output_path)
            print(f"Saved as {output_path}")
        return selected_species
    
    def get_species_df(self, species = 'Prevotella sp. Smarlab 121567', output_dir = "Data/by_species", plot_fig = True, save_fig = True) -> Union[list, pd.DataFrame]:
        """"
        Returns the aggregated dataframe containing the timeseries
        for the desired species across all subjects.
        Saves dataframe to .csv file "Data/by_species/[species name].csv"
        Columns are [day, mouse_1, mouse_2, mouse_3, ... , mouse_N]
        """""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not species in self.species_list:
            raise ValueError("Species name is invalid")
        day_max = int(self.mice_df[0].columns[-1]) # mouse 1 lives the longest
        days = np.arange(day_max + 1).T
        mouse_df_list = []
        mouse_idxs_list = [] #list of subjects indices for which the species is sampled
        for n in range(self.subjects):
            species_data = []
            df = self.mice_df[n].copy()
            df = df[df['species'] == species].iloc[:, 1:] # skip columns [species]
            #check if measures were made that day
            if not df.empty:
                for day in days:
                    if not day in df.columns  or df[day].empty:
                        species_data.append({'day': day, f'mouse_{n + 1}': np.nan})
                    else:
                        species_data.append({'day': day, f'mouse_{n + 1}': df[day].iloc[0]})
            # Handle the case where the species is not sampled for mouse n
            else:
                for day in days:
                    species_data.append({'day': day, f'mouse_{n + 1}': 0})
            mouse_df_list.append(pd.DataFrame(species_data))
            mouse_idxs_list.append(n + 1)
        mouse_columns = [f'mouse_{a}' for a in mouse_idxs_list]
        species_df = mouse_df_list[0]
        for n in range(1,len(mouse_df_list)):
            species_df = pd.concat([species_df, mouse_df_list[n]], axis=1)
        # Drop all but the first occurrence of the 'day' column
        species_df = species_df.loc[:, ~species_df.columns.duplicated()]
        species_df['mean'] = species_df.loc[:, mouse_columns].mean(axis=1, skipna=True)
        species_df['std'] = species_df.loc[:, mouse_columns].std(axis=1, skipna=True)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{species}.csv")
        species_df.to_csv(output_path)
        return species_df


    def plot_species_NEW(self,  species = 'Prevotella sp. Smarlab 121567',  subjects: list = None, figsize = (5,4), save_fig = False, output_path = None) -> None:
        """
        New version with improvements. The old version still works.
        This version can be used to plot ONE SPECIES only,
        with mean and std calculated across all subjects, 
        and a custom number of individual subject lines 
        (plotting for all mice is not suggested since plot becomes too cluttered)
        """
        #assert (isinstance(species_list, list)), "Species must be a list, like [Prevotella, Clostridium, Lactobacillus]"
        
        if subjects is None:
            subjects = list(range(1, self.subjects + 1))
        else:
            assert (isinstance(subjects, list)), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"
            assert all(isinstance(n, int) for n in subjects), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"
            assert all(n <= self.subjects for n in subjects), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"

        mouse_columns = [f'mouse_{n}' for n in subjects if n < (self.subjects + 1)]
        fig, ax = plt.subplots(figsize = figsize)
        colors = cm.get_cmap('tab10')
        imput_path = os.path.join("Data", "by_species", f"{species}.csv")
        while not os.path.exists(imput_path):
            species_df = self.get_species_df(species)
        species_df = pd.read_csv(imput_path)
        intersected_columns = [col for col in mouse_columns if col in species_df.columns]
        for i, column in enumerate(intersected_columns):
            valid_data_df = species_df[['day', column]].dropna() #
            if not valid_data_df.empty:
                #last_day_index = max(j for j, x in enumerate(species_df[column]) if not np.isnan(x))
                #days_of_life = days[:last_day_index + 1]
                x_known = valid_data_df['day']
                y_known =  valid_data_df[column]
                ax.scatter(x_known, y_known, 
                        label=column,
                        alpha=1,
                        #marker=markers[i % len(markers)],
                        color = colors(i))
                ax.plot(x_known, y_known,
                            alpha=1,
                            linewidth = 0.5,
                            #linestyle=linestyles[i % len(linestyles)],
                            color = colors(i))
                ax.set_xlabel("Day", fontsize = 'large')
                ax.set_ylabel("Reads", fontsize = 'large')
        # Plot mean and std
        #ax.scatter(species_df['day'], species_df['mean'], color = 'black')
        valid_data = species_df[['day', 'mean', 'std']].dropna()
        ax.plot(valid_data['day'], valid_data['mean'], color = 'black', linewidth = 1.5, label = 'mean (all mice)')
        ax.fill_between(x=valid_data['day'], y1 = np.maximum(valid_data['mean'] - valid_data['std'], 0), y2=valid_data['mean'] + valid_data['std'], color='black', alpha=0.25, label = 'std (all mice)')
        ax.set_title(f'{species}',  fontsize = 'x-large')
        ax.legend( fontsize = 'large', loc = 'upper right')
        ax.grid(True)
        plt.tight_layout()
        if save_fig:
            plt.savefig(output_path, dpi = 500)
            print(f"saved as {output_path}")
            plt.close()
        return

    def plot_species_MEAN_ONLY(self,  species_list = ['Prevotella sp. Smarlab 121567'], size = (5,4), save_fig = False, output_path = None) -> None:
        assert (isinstance(species_list, list)), "Species must be a list, like [Prevotella, Clostridium, Lactobacillus]"
        fig, ax = plt.subplots(figsize = size)
        colors = cm.get_cmap('tab10')
        for i,species in enumerate(species_list):
            imput_path = os.path.join("Data", "by_species", f"{species}.csv")
            while not os.path.exists(imput_path):
                species_df = self.get_species_df(species)
            species_df = pd.read_csv(imput_path)
            # Plot mean and std
            valid_data = species_df[['day', 'mean', 'std']].dropna()
            ax.plot(valid_data['day'], valid_data['mean'], linewidth = 1.5, color = colors(i % 10), label = f'{species}')
            ax.fill_between(x=valid_data['day'], y1 = np.maximum(valid_data['mean'] - valid_data['std'], 0), y2=valid_data['mean'] + valid_data['std'], color = colors(i % 10), alpha=0.25)
        ax.set_xlabel("Day", fontsize = 'large')
        ax.set_ylabel("Reads", fontsize = 'large')
        legend = ax.legend( fontsize = 'large', loc = 'upper right')
        for line in legend.get_lines():
            line.set_linewidth(10)
        ax.grid(True)
        plt.tight_layout()
        if save_fig:
            plt.savefig(output_path, dpi = 500)
            print(f"saved as {output_path}")
            plt.close()
        else:
            plt.show()
        return




    def get_dissimilarities(self, mouse, write = True):
        """
        Computes dissimilarities for each species in a given mouse dataset and saves values in a csv file if write = True
        The output is a DataFrame where each row corresponds to a species, and columns are time lags (T).
        Only valid time lags (with computed dissimilarities) will appear as columns.
        """
        if write: 
            output_dir = 'Data/dissimilarities'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'dissimilarity_{mouse}.csv')

        # Select mouse dataframe
        mouse_df = self.mice_df[mouse - 1]
        days = mouse_df.columns[4:].to_numpy(dtype=int)  # Days are from the 5th column onward
        max_T = np.max(days)

        # Initialize a dictionary to store dissimilarities for each species
        dissimilarity_data = {sp: {} for sp in mouse_df['species'].to_numpy()}

        # Precompute which days + T are valid for each day
        valid_day_pairs = {T: {day for day in days if (day + T) in days} for T in range(1, max_T + 1)}

        # For each species, compute dissimilarities by time lag
        for sp_idx, sp in enumerate(mouse_df['species'].to_numpy()):
            for T in range(1, max_T + 1):
                total_diss = 0
                count = 0
                for day in valid_day_pairs[T]:
                    diss_t = ((mouse_df[day][sp_idx] - mouse_df[day + T][sp_idx]) / 
                            max(1, (mouse_df[day][sp_idx] + mouse_df[day + T][sp_idx])))**2
                    total_diss += diss_t
                    count += 1

                # Only store dissimilarities for time lags with valid days
                if count > 0:
                    dissimilarity_data[sp][f'{T}'] = total_diss / count

        # Convert the dissimilarity data dictionary to a DataFrame
        dissimilarity_df = pd.DataFrame.from_dict(dissimilarity_data, orient='index')

        # Save the DataFrame to a CSV file
        if write: 
            dissimilarity_df.to_csv(output_path)
            # Print a message with the output path
            print(f"Dissimilarities saved for mouse {mouse} at {output_path}")
        
        # Return the DataFrame as output
        return dissimilarity_df


def get_dissimilarities_genus(mouse, write = True):
    """
        Computes dissimilarities for each species in a given mouse dataset and saves values in a csv file if write = True
        The output is a DataFrame where each row corresponds to a species, and columns are time lags (T).
        Only valid time lags (with computed dissimilarities) will appear as columns.
    """
    if write: 
        output_dir = 'Data/dissimilarities_genus'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'dissimilarity_{mouse}.csv')

    # Select mouse dataframe
    mouse_df = pd.read_csv(f'Data/by_mouse_genus_aggregated/mouse_{mouse-1}_genus.csv')
    days = mouse_df.columns[1:-2].to_numpy(dtype=int)  # Days are from the 2nd column until last two
    mouse_df.columns = [int(c) if c.isdigit() else c for c in mouse_df.columns]
    max_T = np.max(days)

        # Initialize a dictionary to store dissimilarities for each genus
    dissimilarity_data = {ge: {} for ge in mouse_df['Genus'].to_numpy()}

        # Precompute which days + T are valid for each day
    valid_day_pairs = {T: {day for day in days if (day + T) in days} for T in range(1, max_T + 1)}

        # For each genus, compute dissimilarities by time lag
    for ge_idx, ge in enumerate(mouse_df['Genus'].to_numpy()):
        for T in range(1, max_T + 1):
            total_diss = 0
            count = 0
            for day in valid_day_pairs[T]:
                diss_t = ((mouse_df[day][ge_idx] - mouse_df[day + T][ge_idx]) / 
                            max(1, (mouse_df[day][ge_idx] + mouse_df[day + T][ge_idx])))**2
                total_diss += diss_t
                count += 1

            # Only store dissimilarities for time lags with valid days
            if count > 0:
                dissimilarity_data[ge][f'{T}'] = total_diss / count

        # Convert the dissimilarity data dictionary to a DataFrame
    dissimilarity_df = pd.DataFrame.from_dict(dissimilarity_data, orient='index')

    # Save the DataFrame to a CSV file
    if write: 
        dissimilarity_df.to_csv(output_path)
        # Print a message with the output path
        print(f"Dissimilarities for genus saved for mouse {mouse} at {output_path}")
        
        # Return the DataFrame as output
    return dissimilarity_df

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_dissimilarities_in_pdf(mice_diss, output_dir=r"Inspection_Outputs\dissimilarityplots", n_species_per_plot=5, window_size=10, ma = True):
    
    #This function takes as inputs: 
    #    mice_diss: list of dissimilarity data_frames for each mouse
    #    output_dir: directory where you want to save the pdf
    #    n_species_per_plot: how many species you want per plot
    #    if ma = True: you're computing the moving average, you should also indicate the window_size for the average
    
    if os.path.exists(output_dir) and any(f.endswith(".pdf") for f in os.listdir(output_dir)):
        print(f"PDF files already exist in {output_dir}. Delete directory {output_dir} to regenerate plots.")
        return  # Stop execution if any PDFs are found
    else:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        for mouse_idx, df in enumerate(mice_diss):  # Iterate through each mouse's dataframe
            pdf_path = os.path.join(output_dir, f"dissimilarity_{mouse_idx + 1}_plot.pdf")
            
            abs_pdf_path = os.path.abspath(pdf_path)

            print(f"Checking existence of: {abs_pdf_path}")
            
            if os.path.exists(abs_pdf_path):
                print(f"Pdf already exists: {abs_pdf_path}. Skipping.")
                continue  # Skip if file exists

            print(f"Generating and saving: {abs_pdf_path}")
            
            species = np.asarray(df.index)  # Get all species
            lags = np.asarray(df.columns)  # Get lags
            if not os.path.exists(pdf_path):  
                with PdfPages(pdf_path) as pdf:
                    for i in range(0, len(species), n_species_per_plot):
                        selected_species = species[i:i + n_species_per_plot]
                        plt.figure(figsize=(10, 6))
                        for sp in selected_species:
                            data = df.loc[sp].values
                            if ma: 
                                smoothed_data = moving_average(data, window_size=window_size)
                                plt.plot(lags[:len(smoothed_data)], smoothed_data, label=sp)
                            else: 
                                plt.plot(lags, data, label=sp)
                        
                        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.xlabel("Lags")
                        plt.ylabel("Smoothed Values")
                        plt.title(f"Mouse {mouse_idx + 1}: Species {i+1} to {min(i+n_species_per_plot, len(species))}")
                        pdf.savefig()
                        plt.close()
            print(f"Plots saved in {pdf_path}")
    return 

def plot_dissimilarities_genus_in_pdf(mice_diss, output_dir=r"Inspection_Outputs\dissimilarityplots_genus", n_genus_per_plot=5, window_size=10, ma = True):
    
    #This function takes as inputs: 
    #    mice_diss: list of dissimilarity data_frames for each mouse
    #    output_dir: directory where you want to save the pdf
    #    n_genus_per_plot: how many genuses you want per plot
    #    if ma = True: you're computing the moving average, you should also indicate the window_size for the average
    
    if os.path.exists(output_dir) and any(f.endswith(".pdf") for f in os.listdir(output_dir)):
        print(f"PDF files already exist in {output_dir}. Delete directory {output_dir} to regenerate plots.")
        return  # Stop execution if any PDFs are found
    else:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        for mouse_idx, df in enumerate(mice_diss):  # Iterate through each mouse's dataframe
            pdf_path = os.path.join(output_dir, f"dissimilarity_{mouse_idx + 1}_plot.pdf")
            
            abs_pdf_path = os.path.abspath(pdf_path)

            print(f"Checking existence of: {abs_pdf_path}")
            
            if os.path.exists(abs_pdf_path):
                print(f"Pdf already exists: {abs_pdf_path}. Skipping.")
                continue  # Skip if file exists

            print(f"Generating and saving: {abs_pdf_path}")
            
            genuses = np.asarray(df.index)  # Get all genuses
            lags = np.asarray(df.columns)  # Get lags
            if not os.path.exists(pdf_path):  
                with PdfPages(pdf_path) as pdf:
                    for i in range(0, len(genuses), n_genus_per_plot):
                        selected_genuses = genuses[i:i + n_genus_per_plot]
                        plt.figure(figsize=(10, 6))
                        for ge in selected_genuses:
                            data = df.loc[ge].values
                            if ma: 
                                smoothed_data = moving_average(data, window_size=window_size)
                                plt.plot(lags[:len(smoothed_data)], smoothed_data, label=ge)
                            else: 
                                plt.plot(lags, data, label=ge)
                        
                        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.xlabel("Lags")
                        plt.ylabel("Dissimilarity (Smoothed Values)")
                        plt.title(f"Mouse {mouse_idx + 1}: Genus {i+1} to {min(i+n_genus_per_plot, len(genuses))}")
                        pdf.savefig()
                        plt.close()
            print(f"Plots saved in {pdf_path}")
    return 


def plot_dissfit_in_pdf(mice_diss, output_dir=r"Inspection_Outputs\dissimilarityfit", n_species_per_plot=5):
    """
    This function takes as inputs: 
        mice_diss: list of dissimilarity data_frames for each mouse
        output_dir: directory where you want to save the pdf
        n_species_per_plot: how many species you want per plot
    """

    if os.path.exists(output_dir) and any(f.endswith(".pdf") for f in os.listdir(output_dir)):
        print(f"PDF files already exist in {output_dir}. Delete directory {output_dir} to regenerate plots.")
        return  # Stop execution if any PDFs are found
    else:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        for mouse_idx, mouse_df in enumerate(mice_diss):  # Iterate through each mouse's dataframe
            pdf_path = os.path.join(output_dir, f"dissimilarity_{mouse_idx + 1}_fit.pdf")
            species = np.asarray(mouse_df.index)  # Get all species
            lags = np.asarray(mouse_df.columns, dtype = int)  # Get lags

            with PdfPages(pdf_path) as pdf:
                for i in range(0, len(species), n_species_per_plot):
                    selected_species = species[i:i + n_species_per_plot]

                    plt.figure(figsize=(10, 6))
                    for sp in selected_species:
                        data = mouse_df.loc[sp].values
                        m, q, *_ = linregress(lags, data)
                        pred_data = m*lags + q
                        plt.scatter(lags, data, s = 0.3, label = sp)
                        plt.plot(lags, pred_data)

                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.xlabel("Time Lags")
                    plt.ylabel("Dissimilarity")
                    plt.title(f"Mouse {mouse_idx + 1}: dissimilarities from species {i+1} to {min(i+n_species_per_plot, len(species))}")
                    pdf.savefig()
                    plt.close()
            
            print(f"Plots saved in {pdf_path}")
        return 

def plot_dissfit_genus_in_pdf(mice_diss, output_dir=r"Inspection_Outputs\dissimilarityfit_genus", n_genus_per_plot=5):
    """
    This function takes as inputs: 
        mice_diss: list of dissimilarity data_frames for each mouse
        output_dir: directory where you want to save the pdf
        n_genus_per_plot: how many genuses you want per plot
    """

    if os.path.exists(output_dir) and any(f.endswith(".pdf") for f in os.listdir(output_dir)):
        print(f"PDF files already exist in {output_dir}. Delete directory {output_dir} to regenerate plots.")
        return  # Stop execution if any PDFs are found
    else:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        for mouse_idx, mouse_df in enumerate(mice_diss):  # Iterate through each mouse's dataframe
            pdf_path = os.path.join(output_dir, f"dissimilarity_{mouse_idx + 1}_fit.pdf")
            genuses = np.asarray(mouse_df.index)  # Get all genuses
            lags = np.asarray(mouse_df.columns, dtype = int)  # Get lags

            with PdfPages(pdf_path) as pdf:
                for i in range(0, len(genuses), n_genus_per_plot):
                    selected_genus = genuses[i:i + n_genus_per_plot]

                    plt.figure(figsize=(10, 6))
                    for ge in selected_genus:
                        data = mouse_df.loc[ge].values
                        m, q, *_ = linregress(lags, data)
                        pred_data = m*lags + q
                        plt.scatter(lags, data, s = 0.3, label = ge)
                        plt.plot(lags, pred_data)

                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.xlabel("Time Lags")
                    plt.ylabel("Dissimilarity")
                    plt.title(f"Mouse {mouse_idx + 1}: dissimilarities from genus {i+1} to {min(i+n_genus_per_plot, len(genuses))}")
                    pdf.savefig()
                    plt.close()
            
            print(f"Plots saved in {pdf_path}")
        return 

def params_to_npz(mice_ab, objective, bounds, LM, find_transient_end, file_path="parameters_data.npz", force_recompute=False):
    '''
    This function fits the abundancies with a Logistic Model, 
    finds the end of transient with the function 'find_transient_end',
    computes theoretical estimates of K and sigma, saves them in a npz file
    and plots the real data and the fitted LM.
    '''
    if not os.path.exists(file_path) or force_recompute:  
        num_mice = 8
        num_genus = max(len(df['Genus']) for df in mice_ab)
        parameters = np.zeros((num_mice, num_genus, 3))

        for mouse_num, df in enumerate(mice_ab):  # Loop over all mice
            genuses = df['Genus']

            with PdfPages(f"Mouse_{mouse_num+1}_LM_Fits.pdf") as pdf:
                fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
                plot_count = 0  # Counter to track species in a single figure
            
                for idx, ge in enumerate(genuses):  # Loop over all species in the mouse
                    ts_data = df.iloc[idx, 1:].to_numpy()
                    nonzero = np.nonzero(ts_data)[0]

                    if nonzero.size > 0:  # Ensure there is at least one nonzero value
                        start_idx = nonzero[0] + 1  # Adjust for skipped species column
                        time_series = df.iloc[idx, start_idx:].to_numpy()  
                        days = np.asarray(df.columns[start_idx:], dtype=int)  
                        dt = np.diff(days)  
                        result = opt.differential_evolution(objective, bounds, args=(time_series, dt), strategy='best1bin')
                        K_opt, tau_opt = result.x

                        # Simulate data using optimized parameters
                        fitted_data = LM(time_series[0], K_opt, tau_opt, len(time_series)-1, dt)

                        # Identify transient phase: Discard data until it reaches carrying capacity
                        transient_end_idx = find_transient_end(fitted_data)
                    
                        # Keep only post-transient data
                        time_series_red = time_series[transient_end_idx:]

                        # Compute theoretical estimates of K and sigma
                        var = np.var(time_series_red)
                        mean = np.mean(time_series_red)
                        sigma_th = 2 * var / (mean**2 + var)
                        K_th = mean * 2 / (2 - sigma_th)

                        # Store computed values
                        parameters[mouse_num, idx, 0] = K_th
                        parameters[mouse_num, idx, 1] = tau_opt
                        parameters[mouse_num, idx, 2] = sigma_th

                        # Plot current genus in the same figure
                        ax.scatter(days, time_series, marker='o', s=30, label=f"Genus {ge} (Real)")
                        ax.plot(days, fitted_data, linestyle='--', label=f"Genus {ge} (Fitted)")

                        plot_count += 1
                        if plot_count == 5:
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Abundance")
                            ax.set_xlim(0, 100)
                            ax.legend()
                            ax.set_title(f"Mouse {mouse_num+1} - Genus Fit")
                            
                            pdf.savefig(fig)  # Save current figure to PDF
                            plt.close(fig)  # Close the current figure
                            
                            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure
                            plot_count = 0  # Reset counter

                    else:  # If genus is always zero
                        parameters[mouse_num, idx, :] = [0, 0, 0]

                # Save any remaining plots (if less than 5 genus were plotted in the last figure)
                if plot_count > 0:
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Abundance")
                    ax.set_xlim(0, 100)
                    ax.legend()
                    ax.set_title(f"Mouse {mouse_num+1} - Genus Fit")
                    pdf.savefig(fig)  
                    plt.close(fig)  

            print(f"Saved PDF with LM fit for Mouse {mouse_num+1}")

        # Save parameters
        print("Shape of parameters array:", parameters.shape)  # Should be (num_mice, num_genus, 3)
        np.savez(file_path, parameters=parameters)
        print(f"Parameters saved in {file_path}")

    else:
        loaded_data = np.load(file_path)
        parameters = loaded_data['parameters']  # Retrieve the 'parameters' array
        print("Parameters loaded from file")
    
    return parameters



def params_to_csv(mice_ab, objective, bounds, LM, find_transient_end, force_recompute=False):
    """
    This function processes the abundance data for multiple mice, fits the logistic model 
    to each genus' time series using optimization, computes theoretical estimates for the carrying 
    capacity (K), tau (time constant), and sigma (variance-related parameter), and saves the results 
    in CSV files.

    It checks if the CSV files for K, tau, and sigma already exist. If they do, it loads them unless 
    the `force_recompute` flag is set to True, in which case the computation is redone. The function 
    also generates PDF plots showing the model fit for each genus and mouse.

    Args:
    - mice_ab: A list of DataFrames containing the species abundance time series for each mouse.
    - objective: The objective function for the optimization (e.g., the logistic model error).
    - bounds: The bounds for the optimization parameters (K and tau).
    - LM: The logistic model function to simulate abundance over time.
    - find_transient_end: A function to determine where the transient phase ends in the simulated data.
    - force_recompute (bool, optional): If set to True, forces recomputation even if the CSV files exist. Default is False.
    
    Returns:
    - K_df: DataFrame of carrying capacity estimates (K) for each genus and mouse.
    - tau_df: DataFrame of tau (time constant) estimates for each genus and mouse.
    - sigma_df: DataFrame of sigma (variance-related) estimates for each genus and mouse.
    """
    
    file_path_K = "K_values.csv"
    file_path_tau = "tau_values.csv"
    file_path_sigma = "sigma_values.csv"

    # Check if the CSV files already exist and if recomputation is required
    if not os.path.exists(file_path_K) or not os.path.exists(file_path_tau) or not os.path.exists(file_path_sigma) or force_recompute:
        # Initialize dictionaries to store parameters per mouse
        K_dict = {}
        tau_dict = {}
        sigma_dict = {}

        all_genus = sorted(set(sp for df in mice_ab for sp in df['Genus']))  # Get all unique genus

        # Loop over all mice
        for mouse_num in range(8):  
            df = mice_ab[mouse_num]
            genus = df['Genus']

            # Initialize dictionaries for this mouse (default to NaN)
            K_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_genus}
            tau_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_genus}
            sigma_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_genus}

            with PdfPages(f"Inspection_Outputs\LMfits_genus\Mouse_{mouse_num+1}_LM_Fits.pdf") as pdf:
                fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
                plot_count = 0  # Counter to track species in a single figure

                for idx, ge in enumerate(genus):  # Loop over all species in the mouse
                    ts_data = df.iloc[idx, 1:].to_numpy()
                    nonzero = np.nonzero(ts_data)[0]

                    if nonzero.size > 0:  # Ensure there is at least one nonzero value
                        start_idx = nonzero[0] + 1  # Adjust for skipped species column
                        time_series = df.iloc[idx, start_idx:].to_numpy()
                        days = np.asarray(df.columns[start_idx:], dtype=int)
                        dt = np.diff(days)
                        result = opt.differential_evolution(objective, bounds, args=(time_series, dt), strategy='best1bin')
                        K_opt, tau_opt = result.x

                        # Simulate data using optimized parameters
                        fitted_data = LM(time_series[0], K_opt, tau_opt, len(time_series)-1, dt)

                        # Identify transient phase: Discard data until it reaches carrying capacity
                        transient_end_idx = find_transient_end(fitted_data)

                        # Keep only post-transient data
                        time_series_red = time_series[transient_end_idx:]

                        # Compute theoretical estimates of K and sigma
                        var = np.var(time_series_red)
                        mean = np.mean(time_series_red)
                        sigma_th = 2 * var / (mean**2 + var)
                        K_th = mean * 2 / (2 - sigma_th)

                        # Store computed values in dictionaries
                        K_dict[f"Mouse {mouse_num+1}"][ge] = K_th
                        tau_dict[f"Mouse {mouse_num+1}"][ge] = tau_opt
                        sigma_dict[f"Mouse {mouse_num+1}"][ge] = sigma_th

                        # Plot current species in the same figure
                        ax.scatter(days, time_series, marker='o', s=30, label=f"Genus {ge} (Real)")
                        ax.plot(days, fitted_data, linestyle='--', label=f"Genus {ge} (Fitted)")

                        plot_count += 1
                        if plot_count == 5:
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Abundance")
                            ax.set_xlim(0, 100)
                            ax.legend()
                            ax.set_title(f"Mouse {mouse_num+1} - Genus Fit")
                            
                            pdf.savefig(fig)  # Save current figure to PDF
                            plt.close(fig)  # Close the current figure
                            
                            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure
                            plot_count = 0  # Reset counter

                    else:  # If species is always zero, store 0 values
                        K_dict[f"Mouse {mouse_num+1}"][ge] = 0
                        tau_dict[f"Mouse {mouse_num+1}"][ge] = 0
                        sigma_dict[f"Mouse {mouse_num+1}"][ge] = 0
            print(f"Saved PDF with LM fit for Mouse {mouse_num+1}")

        # Convert dictionaries to DataFrames
        K_df = pd.DataFrame(K_dict)
        tau_df = pd.DataFrame(tau_dict)
        sigma_df = pd.DataFrame(sigma_dict)

        # Save DataFrames to CSV files
        K_df.to_csv(file_path_K)
        tau_df.to_csv(file_path_tau)
        sigma_df.to_csv(file_path_sigma)
        print("Parameters dataframes saved in csv files")

        # Display DataFrames
        print("K Values DataFrame:")
        print(K_df)

        print("Tau Values DataFrame:")
        print(tau_df)

        print("Sigma Values DataFrame:")
        print(sigma_df)

    else:
        print('PDF files already exist in Inspection_Outputs\LMfits. Delete directory Inspection_Outputs\LMfits to regenerate plots.')
        K_df = pd.read_csv(file_path_K)
        tau_df = pd.read_csv(file_path_tau)
        sigma_df = pd.read_csv(file_path_sigma)
        print('Loaded K, sigma, tau dataframes from csv files.')

    return K_df, tau_df, sigma_df


extended_tab20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
    "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c",
    "#7b4173", "#a55194", "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6",
    "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
    "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8",
    "#bcbddc", "#dadaeb", "#636363", "#969696", "#bdbdbd", "#d9d9d9",
    "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2",
    "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf",
    "#9edae5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f"
]
