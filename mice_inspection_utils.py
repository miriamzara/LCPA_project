import random
from typing import Union
import numpy as np
import seaborn as sns
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
from scipy.optimize import minimize
import math
from matplotlib.backends.backend_pdf import PdfPages

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
            df.rename(columns= {col: int(col) for col in df.columns[4:]}, inplace=True)
            # Sort values
            if sort_by == 'mean':
                df.sort_values(by='mean_counts', ascending=False, inplace=True)
            else:
                df.sort_values(by='median_counts', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            mice_df.append(df)
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
            ######### delete multiple measures on the same day ###########
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
            ##########################################            
            # group otus corresponding to same species
            ##########################################
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
            ###########################################
            # Add columns with mean counts and median counts
            df.insert(1, "mean_counts", df.iloc[:, 1:].mean(axis=1))
            df.insert(1, "median_counts", df.iloc[:, 1:].median(axis=1))
            ###########################################
            # Sort values
            df.sort_values(by='mean_counts', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            # Save output
            ###########################################
            df.to_csv(op[i])
        return

    def transpose_mice_df(self) -> None:
        """
        Transposes the mice dataframes and fills missing days with NaN values.
        Creates CSV outputs with day as the first column and species in subsequent columns.
        """
        for n in range(self.subjects):
            df =  self.mice_df[n].copy()
            days = df.columns[4:].to_list() 
            df.drop(columns=['median_counts', 'mean_counts', 'Unnamed: 0'], inplace = True)
            df = df.set_index('species').transpose()
            df.reset_index(inplace = True, drop = True)
            df.insert(0, 'day', days)
            full_day_range = pd.Series(range(df['day'].min(), df['day'].max() + 1))
            df.set_index('day', inplace=True)
            df = df.reindex(full_day_range)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'day'}, inplace=True)
            if not os.path.exists('Data'):
                os.makedirs('Data')
            output_path = os.path.join("Data", f"mouse_{n + 1}_transposed.csv")
            df.to_csv(output_path)
        return 

    def interpolate_mice_df(self, max_rank = 4):
        selected_species = self.select_species(max_rank)
        interpolated_df_list = []
        for subject in range(self.subjects):
            imput_path = f'Data/mouse_{subject + 1}_transposed.csv'
            output_path = f'Data/mouse_{subject + 1}_splines.csv'
            if not os.path.exists(imput_path):
                self.transpose_mice_df()
            df = pd.read_csv(imput_path)
            days_of_life = list(df['day'])
            valid_data_df = df.dropna()
            days_with_measures = list(valid_data_df['day'])

            interpolated_df = pd.DataFrame(days_of_life, columns=["day"])

            for species in df.columns[2:]:
                if species in selected_species:
                    y_known =  list(valid_data_df[species])
                    spline = CubicSpline(days_with_measures, y_known)
                    #spline = non_neg_spline(days_with_measures, y_known)
                    y_interpolated = spline(days_of_life)
                    y_interpolated = np.clip(y_interpolated, a_min= 0.5, a_max = None) # should be superfluous, but to be sure
                    interpolated_df.insert(len(interpolated_df.columns), column = species, value = y_interpolated)
            interpolated_df.drop(columns= 'day', inplace= True)
            # Sort the columns alphabetically
            sorted_columns = sorted(interpolated_df.columns)
            interpolated_df = interpolated_df[sorted_columns]
            interpolated_df_list.append(interpolated_df)
            interpolated_df.to_csv(output_path)
        return interpolated_df_list


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
    


    def plot_stackedbar(self, max_rank, category, by = 'mean', save_fig = True) -> None:
        """
        Makes stacked bar plot of counts aggregated by cathegory and saves pdf plots
        as "Inspection_Outputs/Stacked/{category}_stacked_{max_rank}_by_{by}.pdf"
        """

        categories = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
        assert isinstance(max_rank, int) and max_rank > 2, "Max rank must be the number of most abundant species considered"
        assert category in categories, "Cathegory must be one of Phylum, Class, Order, Family, Genus, Species"
        assert by in ['median', 'mean'], "by must be 'median' or 'mean'"

        if save_fig:
            dir = os.path.join(self.output_dir, "Stacked")
            os.makedirs(dir, exist_ok=True)
            output_path = os.path.join(dir, f"{category}_stacked_{max_rank}_by_{by}.pdf")


        selected_species = self.select_species(max_rank= max_rank, by= by)
        stacked_df = self.metadata_df[["Phylum", "Class", "Order", "Family", "Genus", "Species"]].drop_duplicates()
        stacked_df = stacked_df[stacked_df['Species'].isin(selected_species)]
        stacked_df.sort_values(by='Species', inplace=True) #re-arranges species from A to Z
        for i in range(self.subjects):
            frequencies = []
            frequency = 0
            for species in selected_species:
                temp = self.mice_df[i][self.mice_df[i]["species"] == species]
                if not temp.empty:
                    frequency = temp["mean_counts"].iloc[0]/self.sample_size
                frequencies.append(frequency)
            stacked_df.insert(len(stacked_df.columns), f"mouse_{i + 1}", frequencies)
        if category == 'Species':
            stacked_df.drop(columns= ["Phylum", "Class", "Order", "Family", "Genus"], inplace=True)
            stacked_df.sort_values(by='mouse_1', ascending = False, inplace=True) #re-arranges for better visualization
            weight_counts = {}
            for i, row in stacked_df.iterrows():
                name = row['Species']
                values = row[[f"mouse_{j+ 1}" for j in range(self.subjects)]].tolist()
                weight_counts[name] = values
        else:
            category_index = categories.index(category)
            cols_to_delete = np.delete(categories, category_index)
            stacked_df.drop(columns= cols_to_delete, inplace=True)
            results = stacked_df.groupby(category, as_index=False).sum()
            results.sort_values(by='mouse_1', ascending = False, inplace=True) # better visualization
            weight_counts = {}
            for i, row in results.iterrows():
                name = row[category]
                values = row[[f"mouse_{j+ 1}" for j in range(self.subjects)]].tolist()
                weight_counts[name] = values
        ###############
        colors =  extended_tab20
        width = 0.5
        x_labels = [f"{j + 1}" for j in range(self.subjects)]
        fig = plt.figure(dpi=300)
        gs = gridspec.GridSpec(nrows = 1, ncols= 6, figure=fig)
        # Create the subplots with different dimensions
        ax1 = fig.add_subplot(gs[:, :3])
        ax2 = fig.add_subplot(gs[:, 3:])  
        bottom = np.zeros(self.subjects)
        for idx, (species_name, weight_count) in enumerate(weight_counts.items()):
            ax1.bar(x_labels, weight_count, width, label= species_name, bottom=bottom, color = colors[idx % 20])
            bottom += weight_count
        ax1.set_ylabel("Mean frequency")
        ax1.set_xlabel("mouse")
        max_items_per_col = 40  # Set this according to how many items per column you'd like
        ncol = math.ceil(len(weight_counts.items()) / max_items_per_col)
        ax1.legend( title = f"Most Abundant {category}",
                    bbox_to_anchor=(+2.2, 0.5),
                    loc='right', ncol=ncol, handlelength=2,
                    handleheight=1,
                    fontsize = 'xx-small',
                    title_fontsize = 'x-small')
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.grid()
        ax2.axis("off")
        plt.suptitle(f"{category} Abundances (threshold: first {max_rank} species)")
        #################
        if save_fig:
            plt.savefig(output_path)
        return 

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
            df = df[df['species'] == species].iloc[:, 4:] # skip columns [unnamed, species, mean, median]
            #check if species is sampled for mouse n
            if not df.empty:
                for day in days:
                    if not day in df.columns  or df[day].empty:
                        species_data.append({'day': day, f'mouse_{n + 1}': 0})
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

    def plot_species(self, interpolate: str = None, species_list = ['Prevotella sp. Smarlab 121567'],  subjects: list = None,  save_fig = False) -> None:
        """
        Makes time series plots. If save_fig is set to True,
        plot is saved as "Inspection_Outputs/[num_species_across_page_X or name_across or name_mouse].pdf"
        Arguments
        - species: list of species names. each species gets a separate subplot.
        - subjects: list of subject indeces.
        """
        assert (isinstance(species_list, list)), "Species must be a list, like [Prevotella, Clostridium, Lactobacillus]"
        
        if subjects is None:
            subjects = list(range(1, self.subjects + 1))
        else:
            assert (isinstance(subjects, list)), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"
            assert all(isinstance(n, int) for n in subjects), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"
            assert all(n <= self.subjects for n in subjects), f"Subjects must be a list of integers in the range 1 (included) to {len(self.subjects)} (included)"

        if interpolate is not None:
            assert interpolate in ['splines'], "parameter 'interpolate' must be 'splines' if selected"

        if save_fig:
            output_dir = "Inspection_Outputs"
            dir = os.path.join(self.output_dir, "Time_Series")
            os.makedirs(dir, exist_ok=True)
            output_path_beginning = os.path.join(output_dir, f"{len(species_list)}_species_{len(subjects)}_subjects")
    
        mouse_columns = [f'mouse_{n}' for n in subjects if n < (self.subjects + 1)]

        num_species = len(species_list)
        rows = 5
        cols = 2
        n_pages = math.ceil(num_species /(rows * cols))

        for n in range(n_pages):
            start_idx = n*10 +1
            species_to_plot = species_list[(n* 10):min((n+1)* 10, len(species_list))]
            num_species = len(species_to_plot)
            fig, axes = plt.subplots(rows, cols, figsize= (21.0, (29.7/ 5) * rows) ,dpi= 300, squeeze=False)
            axes = axes.flatten() 
            for j in range(num_species, len(axes)):
                axes[j].axis('off')
            for s, species in enumerate(species_to_plot):
                imput_path = os.path.join("Data", "by_species", f"{species}.csv")
                while not os.path.exists(imput_path):
                    species_df = self.get_species_df(species)
                species_df = pd.read_csv(imput_path)
                markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
                linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
                colors = cm.get_cmap('tab10')
                days = list(species_df['day'])
                intersected_columns = [col for col in mouse_columns if col in species_df.columns]
                for i, column in enumerate(intersected_columns):
                    valid_data_df = species_df[['day', column]].dropna()
                    if not valid_data_df.empty:
                        last_day_index = max(j for j, x in enumerate(species_df[column]) if not np.isnan(x))
                        days_of_life = days[:last_day_index + 1]
                        x_known = valid_data_df['day']
                        y_known =  valid_data_df[column]
                        if interpolate == 'splines':
                            spline = CubicSpline(x_known, y_known)
                            y_interpolated = np.clip(a = spline(days_of_life), a_min= 0.5, a_max= self.sample_size)
                            axes[s].scatter(days_of_life, y_interpolated,
                                        alpha=0.3,
                                        marker=markers[i % len(markers)],
                                        color = colors(i))
                            axes[s].plot(days_of_life, y_interpolated,
                                        alpha=0.3,
                                        linewidth = 0.5,
                                        linestyle=linestyles[i % len(linestyles)],
                                        color = colors(i))
                        else:
                            axes[s].scatter(species_df['day'], species_df[column], 
                                    label=column,
                                    alpha=1,
                                    marker=markers[i % len(markers)],
                                    color = colors(i))
                            axes[s].plot(x_known, y_known,
                                        alpha=1,
                                        linewidth = 0.5,
                                        linestyle=linestyles[i % len(linestyles)],
                                        color = colors(i))
                if len(subjects) > 1:
                    axes[s].scatter(species_df['day'], species_df['mean'], color = 'black')
                    valid_data = species_df[['day', 'mean', 'std']].dropna()
                    axes[s].plot(valid_data['day'], valid_data['mean'], color = 'black', label = 'mean')
                    axes[s].fill_between(x=valid_data['day'], y1 = np.maximum(valid_data['mean'] - valid_data['std'], 0), y2=valid_data['mean'] + valid_data['std'], color='black', alpha=0.4, label = 'std')
                if i // cols == rows - 1: 
                    axes[s].set_xlabel('Days', fontsize = 'large')
                else:
                    axes[s].set_xlabel('') 
                if i % cols == 0:  
                    axes[s].set_ylabel('Counts',  fontsize = 'large')
                else:
                    axes[s].set_ylabel('')  # Hide y-axis label
                axes[s].set_title(f'{s + start_idx}: {species}',  fontsize = 'x-large')
                axes[s].legend( fontsize = 'large', loc = 'upper right')
                axes[s].grid(True)
            plt.tight_layout(pad = 0.5, h_pad= 1, w_pad= 1) #space between edges of figure and edges of sublots
            if save_fig:
                output_path = f"{output_path_beginning}_page_{n + 1}.pdf"
                plt.savefig(output_path)
                print(f"saved as {output_path}")
                plt.close()
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

def params_to_pdf(mi, mice_ab, objective, bounds, LM, find_transient_end, file_path="parameters_data.npz", force_recompute=False):
    if not os.path.exists(file_path) or force_recompute:  
        num_mice = len(mi.mice_df)
        num_species = max(len(df['species']) for df in mice_ab)
        parameters = np.zeros((num_mice, num_species, 3))

        for mouse_num, df in enumerate(mice_ab):  # Loop over all mice
            species = df['species']

            with PdfPages(f"Mouse_{mouse_num+1}_LM_Fits.pdf") as pdf:
                fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
                plot_count = 0  # Counter to track species in a single figure
            
                for idx, sp in enumerate(species):  # Loop over all species in the mouse
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

                        # Plot current species in the same figure
                        ax.scatter(days, time_series, marker='o', s=30, label=f"Species {sp} (Real)")
                        ax.plot(days, fitted_data, linestyle='--', label=f"Species {sp} (Fitted)")

                        plot_count += 1
                        if plot_count == 5:
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Abundance")
                            ax.set_xlim(0, 100)
                            ax.legend()
                            ax.set_title(f"Mouse {mouse_num+1} - Species Fit")
                            
                            pdf.savefig(fig)  # Save current figure to PDF
                            plt.close(fig)  # Close the current figure
                            
                            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure
                            plot_count = 0  # Reset counter

                    else:  # If species is always zero
                        parameters[mouse_num, idx, :] = [0, 0, 0]

                # Save any remaining plots (if less than 5 species were plotted in the last figure)
                if plot_count > 0:
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Abundance")
                    ax.set_xlim(0, 100)
                    ax.legend()
                    ax.set_title(f"Mouse {mouse_num+1} - Species Fit")
                    pdf.savefig(fig)  
                    plt.close(fig)  

            print(f"Saved PDF with LM fit for Mouse {mouse_num+1}")

        # Save parameters
        print("Shape of parameters array:", parameters.shape)  # Should be (num_mice, num_species, 3)
        np.savez(file_path, parameters=parameters)
        print(f"Parameters saved in {file_path}")

    else:
        loaded_data = np.load(file_path)
        parameters = loaded_data['parameters']  # Retrieve the 'parameters' array
        print("Parameters loaded from file")
    
    return parameters

import scipy.optimize as opt

def params_to_csv(mi, mice_ab, objective, bounds, LM, find_transient_end, force_recompute=False):
    """
    This function processes the species abundance data for multiple mice, fits the logistic model 
    to each species' time series using optimization, computes theoretical estimates for the carrying 
    capacity (K), tau (time constant), and sigma (variance-related parameter), and saves the results 
    in CSV files.

    It checks if the CSV files for K, tau, and sigma already exist. If they do, it loads them unless 
    the `force_recompute` flag is set to True, in which case the computation is redone. The function 
    also generates PDF plots showing the model fit for each species and mouse.

    Args:
    - mi: A data structure containing metadata about the mice (e.g., `mi.mice_df`).
    - mice_ab: A list of DataFrames containing the species abundance time series for each mouse.
    - objective: The objective function for the optimization (e.g., the logistic model error).
    - bounds: The bounds for the optimization parameters (K and tau).
    - LM: The logistic model function to simulate abundance over time.
    - find_transient_end: A function to determine where the transient phase ends in the simulated data.
    - force_recompute (bool, optional): If set to True, forces recomputation even if the CSV files exist. Default is False.
    
    Returns:
    - K_df: DataFrame of carrying capacity estimates (K) for each species and mouse.
    - tau_df: DataFrame of tau (time constant) estimates for each species and mouse.
    - sigma_df: DataFrame of sigma (variance-related) estimates for each species and mouse.
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

        num_mice = len(mi.mice_df)
        all_species = sorted(set(sp for df in mice_ab for sp in df['species']))  # Get all unique species

        # Loop over all mice
        for mouse_num in range(len(mi.mice_df)):  
            df = mice_ab[mouse_num]
            species = df['species']

            # Initialize dictionaries for this mouse (default to NaN)
            K_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_species}
            tau_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_species}
            sigma_dict[f"Mouse {mouse_num+1}"] = {sp: np.nan for sp in all_species}

            with PdfPages(f"Mouse_{mouse_num+1}_LM_Fits.pdf") as pdf:
                fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure
                plot_count = 0  # Counter to track species in a single figure

                for idx, sp in enumerate(species):  # Loop over all species in the mouse
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
                        K_dict[f"Mouse {mouse_num+1}"][sp] = K_th
                        tau_dict[f"Mouse {mouse_num+1}"][sp] = tau_opt
                        sigma_dict[f"Mouse {mouse_num+1}"][sp] = sigma_th

                        # Plot current species in the same figure
                        ax.scatter(days, time_series, marker='o', s=30, label=f"Species {sp} (Real)")
                        ax.plot(days, fitted_data, linestyle='--', label=f"Species {sp} (Fitted)")

                        plot_count += 1
                        if plot_count == 5:
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Abundance")
                            ax.set_xlim(0, 100)
                            ax.legend()
                            ax.set_title(f"Mouse {mouse_num+1} - Species Fit")
                            
                            pdf.savefig(fig)  # Save current figure to PDF
                            plt.close(fig)  # Close the current figure
                            
                            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure
                            plot_count = 0  # Reset counter

                    else:  # If species is always zero, store 0 values
                        K_dict[f"Mouse {mouse_num+1}"][sp] = 0
                        tau_dict[f"Mouse {mouse_num+1}"][sp] = 0
                        sigma_dict[f"Mouse {mouse_num+1}"][sp] = 0
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
        K_df = pd.read_csv(file_path_K)
        tau_df = pd.read_csv(file_path_tau)
        sigma_df = pd.read_csv(file_path_sigma)
        print('Loaded K, sigma, tau dataframes from csv files')

    return K_df, tau_df, sigma_df


def format_for_regression(self, subject = 1, max_rank = 10, sampling_interval = 1):
    """""
    Returns the dataframe ready to pass to the Fisher.LIMITS() routine
    Data is interpolated using
    """""
    imput_path = f'Data/mouse_{subject}_transposed.csv'
    if not os.path.exists(imput_path):
        self.transpose_mice_df()
    df = pd.read_csv(imput_path)
    selected_species = self.select_species(max_rank= max_rank)
    if not (isinstance(sampling_interval, int) and sampling_interval > 0):
        raise ValueError("the sampling interval must be an integer number of days")
    regression_df_list = []
    for n in range(self.subjects):
        df =  self.mice_df[n].copy()
        times = df.columns[4:].to_list()
        medians_dict = {}
        means_dict = {}
        for i, row in df.iterrows():
            medians_dict[row['species']] = row['median_counts']
            means_dict[row['species']] = row['mean_counts']
        df.drop(columns=['median_counts', 'mean_counts', 'Unnamed: 0'], inplace = True)
        df = df.set_index('species').transpose()
        df.insert(0, 'day', times)
        columns_to_keep = ['day']
        columns_to_keep.extend(selected_species) #adds columns, names given by elements in list "selected_species"
        df = df[columns_to_keep]
        print(f"mouse {n+1}", df.head())

        #pairs = [(times[i], times[i + 1]) for i in range(len(times) - 1) if times[i + 1] - times[i] == sampling_interval]
        #for pair in pairs:
    """""
    Computes the variables needed to perform linear regression. Deletes all imput_data rows containing near zero values (because log is undefined)
    Stores as numpy nd array  self.data, with nrows = timesteps - 1 - deleted rows , ncols = 2 * num_species
    Caution: some covariate columns may contain inf and nan values. This happens when the species is extinct.
    These rows are deleted when regression over the corresponding species is performed.
    
    #transposed_data = self.input_data[:, :]
    transposed_data = np.transpose(self.input_data[:, :])

    temp = np.log(transposed_data)
    #inf_mask = np.isinf(temp)
    #rows_with_inf = np.any(inf_mask, axis=1)
    #transposed_data = transposed_data[~rows_with_inf]
    #if len(transposed_data) == 0:
    #    raise ValueError("Some species is extinct right from the start. Please compute initial conditions again.")

    log_data = np.zeros(shape= (transposed_data.shape[0] - 1, transposed_data.shape[1]))
    for t in np.arange(0, transposed_data.shape[0] - 1):
        for i in range(transposed_data.shape[1]):
            log_data[t, i] = np.log(transposed_data[t + 1, i]) - np.log(transposed_data[t , i])
    medians = np.median(transposed_data[:, :self.num_species], axis = 0)
    for i, m in enumerate(medians):
        transposed_data[:, i] = transposed_data[:, i] - m
    self.data = np.hstack((transposed_data[:-1, :], log_data))

    """""    
    return 

def non_neg_spline(x, y):
    # Minimize the set of coefficients a_i (aka control points)
    def constraint_func(a):
        cs_constrained = CubicSpline(x, a)
        x_eval = np.linspace(x[0], x[-1], 1000)
        return np.min(cs_constrained(x_eval))
    def objective(a):
        return 0
    constraints = {'type': 'ineq', 'fun': constraint_func}
    initial_guess = y 
    result = minimize(objective, initial_guess, constraints=constraints)
    cs_constrained = CubicSpline(x, result.x)
    print("old a_i =", initial_guess)
    print( "optimized a_i= ", result.x)
    return cs_constrained


def random_color():
    return (random.random(), random.random(), random.random())

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


def bin_counts(data, lower, upper):
    data = np.array(data)
    filtered_data = data[(data >= lower) & (data < upper)]
    count = len(filtered_data)
    return count

#Select subset of times list where consecutive times are separated by at least cutoff days
def extract_subset(time_instants, cutoff):
    subset = [time_instants[0]]
    original_indices_df = [0]
    for i in range(1, len(time_instants)):
        if time_instants[i] - subset[-1] >= cutoff:
            subset.append(time_instants[i])
            original_indices_df.append(i)
    return subset, original_indices_df

"""""
compute Pearson matrix
df: dataframe

returns:
cov: numpy 2d array
cov[n,m] contains pearson coefficient between row n and row m
"""""
def row_pair_correlation(df):
    cov = np.zeros((len(df), len(df)))
    for n, row in df.iterrows():
        for m in range(n, len(df)):
            x = row[3:]
            y = df.iloc[m, 3:]
            cov[n, m] = np.corrcoef(x, y)[0,1] #Pearson coefficient
            cov[m, n] = cov[n, m]
    return cov