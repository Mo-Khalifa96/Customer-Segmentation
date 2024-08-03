#CUSTOMER SEGMENTATION (UNSUPERVISED MACHINE LEARNING FOR CLUSTER ANALYSIS)

#Import modules for use
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, Markdown
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score

import warnings
warnings.simplefilter("ignore")

#Adjust data display options 
pd.set_option('display.max_columns', None)
#Set context for plotting
sns.set_theme(context='paper', style='darkgrid')


#Defining Custom Functions for later use 
#Define function to get color palette for visualization
def get_colors(var, colors):
    if type(colors) == dict: 
        return colors.get(var)
    elif type(colors) in (str, list, tuple): 
        return colors 
    else: 
        return colors.colors
    
#Define function to create and return a scatter plot
def get_scatterplot(x_var: str, y_var: str, clusters_var: str = None, colors: any = None): 
    ax=sns.scatterplot(data=df, x=x_var, y=y_var, hue=clusters_var, palette=colors, s=15, alpha=.75)
    ax.set_title(f'Relationship between {y_var} and {x_var}', fontsize=15)
    ax.set_xlabel(x_var, fontsize=12.5)
    ax.set_ylabel(y_var, fontsize=12.5)
    ax.legend(title=clusters_var, loc='upper right', alignment='left')

#Define function to create and return a boxen plot
def get_boxenplot(x_var: str, y_var: str, title_x: str, title_y: str, clusters_var: str = None, boxen: bool = True, colors: any = None, order: dict = None):
    if boxen:
        ax=sns.boxenplot(data=df, x=x_var, y=y_var, hue=clusters_var if clusters_var != None else x_var, palette=colors, saturation=.9,
                            order=order.get(x_var) if order!= None else None, alpha=.8 if len(np.unique(df[x_var]))<15 else 1, 
                            showfliers=False, width=.5, gap=.25 if len(np.unique(df[x_var])) < 10 else 0)
    else:
        ax=sns.boxplot(data=df, x=x_var, y=y_var, hue=clusters_var if clusters_var != None else x_var, 
                       palette=colors, order=order.get(x_var) if order!= None else None, 
                       width=.8, gap=.25 if len(np.unique(df[x_var])) < 10 else 0)        
        for artist in ax.artists:
            r, g, b, _ = artist.get_facecolor()
            artist.set_facecolor((r, g, b, 0.8))

    if len(np.unique(df[x_var])) > 8: 
        ax.set_aspect(aspect=(.4 if y_var != 'Purchase Amount (USD)' else .25))
    ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15)
    ax.set_xlabel(x_var, fontsize=12.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if (len(ax.get_xticklabels()) > 5 and x_var !='Age Group') else None))
    ax.set_xlim(-1,len(ax.get_xticklabels())+0.2)
    ax.set_ylabel(y_var, fontsize=12.5)
    ax.legend(title=clusters_var, loc='upper right', alignment='left') 

#Define a function to create and return a heatmap 
def get_heatmap(x_var: str, y_var: str, title_x: str, title_y: str, clusters_var: str = None, colors: any = None, order: dict = None):  
    xy_crosstab = pd.crosstab(index=df[y_var], columns=[df[x_var], df[clusters_var]]).reindex((order.get(y_var)[::-1] if (order!= None and y_var in order) else None))        
    if len(xy_crosstab.columns) <= 4:
        xy_crosstab = xy_crosstab.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=[0, 1])
        xy_crosstab = xy_crosstab.reorder_levels([1, 0], axis=1) 
    else:
        xy_crosstab=xy_crosstab.reindex(order.get(x_var,None) if order!= None else None, axis=1, level=0)
        for category in xy_crosstab.columns.levels[0][1:]:
            xy_crosstab[(category, '')] = 0
        xy_crosstab = xy_crosstab.sort_index(axis=1, level=[0, 1]).reindex(order.get(x_var,None) if order!= None else None, axis=1, level=0)
    ax=sns.heatmap(xy_crosstab, ax=plt.gca(),  cmap='gray_r', annot=True, fmt='.0f', alpha=.65, linewidths=.8, annot_kws={'color':'w' if len(xy_crosstab.columns) > 4 else None})
    ax.set_aspect(2 if (len(xy_crosstab.columns) > 4 and len(xy_crosstab.columns) >= len(xy_crosstab.index) and title_x!='Item Purchased') else 'auto')
    ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=(12 if (ax.get_aspect()!='auto' and len(xy_crosstab.columns) > 16) else 15))
    ax.set_xlabel(' — '.join(ax.get_xlabel().split('-')), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12.5)
    xticklabels = ['' if item.get_text().endswith('-') else item for item in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, fontsize=(6 if len(ax.get_xticklabels())>10 else 8))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=(90 if 10 < len(ax.get_yticklabels()) <= 4 else None), fontsize=8)
    colors_dict={'Cluster 1': colors[0], 'Cluster 2': colors[1], 'Cluster 3': colors[2], '': 'w'}
    col_colors = [colors_dict[i] for i in xy_crosstab.columns.get_level_values(1)]
    for i, color in enumerate(col_colors):
        ax.fill_betweenx(y=[0,len(xy_crosstab.index)+1], x1=i, x2=i+1, color=color, alpha=.45)

#Define a function to create and return a pie plot
def get_pieplot(x_var: str, y_var: str, title_x: str, title_y: str, colors: any = None):
    xy_crosstab = pd.crosstab(index=df[x_var], columns=df[y_var])
    flat_crosstab = xy_crosstab.stack().reset_index()
    flat_crosstab.columns = [x_var, y_var, 'Count']
    aggregated_data = flat_crosstab.groupby([y_var, x_var])['Count'].sum().reset_index()
    sizes = aggregated_data['Count']
    labels = [f"{row[x_var]} - {row[y_var]}" for _, row in aggregated_data.iterrows()]

    ax=plt.gca()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[::-1], autopct='%1.1f%%', startangle=100, labeldistance=1.05, frame=True, textprops=dict(fontsize=11), wedgeprops=dict(width=1, linewidth=2.5, edgecolor='lightgray', alpha=.8))
    for i,wedge in enumerate(wedges):
        if i % 2 != 0: 
            continue
        angle = math.radians((wedge.theta1) % 360)
        x, y = math.cos(angle), math.sin(angle)
        line = plt.Line2D([0, 1.01*x], [0, 1.01*y], transform=ax.transData, color='w', linestyle='-', linewidth=3)
        plt.gcf().add_artist(line)
    ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15, pad=13)
    ax.patch.set_facecolor((ax.get_facecolor(),0.77))
    ax.set(xticks=[], yticks=[])
    ax.axis('equal')

#Define a function to create and return a bar plot
def get_barplot(x_var: str, y_var: str, title_x: str, title_y: str, clusters_var: str = None, colors: any = None, order: dict = None):
    if y_var == 'Review Rating':
        ax=sns.barplot(data=df, x=x_var, y=y_var, hue=clusters_var if clusters_var != None else x_var, palette=colors, alpha=.8, saturation=.9, gap=.1, errorbar=None, 
                       width=(.8 if len(np.unique(df[y_var]))>2 else .5), order=order.get(x_var) if order != None else None)
        ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15)
        ax.set_xlabel(x_var, fontsize=12.5)
        ax.set_ylabel(y_var, fontsize=12.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if (len(ax.get_xticklabels()) > 5 and x_var !='Age Group') else None))
        ax.set_ylim(0, 4.5)
    
    else:
        if clusters_var is None:
            color, cmap = None if len(colors) != 2 else colors, colors if len(colors) != 2 else None
            xy_crosstab = pd.crosstab(index=df[y_var], columns=df[x_var]).reindex(order.get(y_var,df[y_var].unique()) if type(order) == dict else df[y_var].unique(), axis=0)
            xy_crosstab = xy_crosstab.reindex(order.get(x_var, df[x_var].unique()) if type(order) == dict else df[x_var].unique(), axis=1, level=0)
            ax=xy_crosstab.plot(kind='bar', ax=plt.gca(), color=color, cmap=cmap, alpha=.8 if len(np.unique(df[x_var])) < 15 else 1, edgecolor='lightgray', rot=0)
            ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15)
            ax.set_xlabel(y_var, fontsize=12.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if (len(ax.get_xticklabels()) > 4 and y_var != 'Age Group') else None))
            ax.set_xlim(-1,len(ax.get_xticklabels())+0.2)
            ax.set_ylabel('Count', fontsize=12.5)

        else:
            if (title_y == 'Gender' and len(np.unique(df[title_x]))>2) or (title_x=='Gender' and len(np.unique(df[title_y]))==2):
                xy_crosstab = pd.crosstab(index=df[y_var], columns=[df[x_var], df[clusters_var]]).reindex(order.get(y_var,df[y_var].unique()) if type(order) == dict else df[y_var].unique(), axis=0) 
                xy_crosstab = xy_crosstab.sort_index(axis=1, level=[0,0]).reindex(order.get(x_var,df[x_var].unique()) if type(order) == dict else df[x_var].unique(), axis=1, level=0)
                colors_dict={'Cluster 1': colors[0], 'Cluster 2': colors[1], 'Cluster 3': colors[2]}
                colors = [colors_dict[i] for i in xy_crosstab.columns.get_level_values(1)]
                ax=xy_crosstab.plot(kind='bar', ax=plt.gca(), color=colors, alpha=.8 if len(np.unique(df[x_var])) < 15 else 1, edgecolor='lightgray', rot=0)
                ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15)
                ax.set_xlabel(y_var, fontsize=12.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if (len(ax.get_xticklabels()) > 4 and y_var != 'Age Group') else None))
                ax.set_xlim(-1,len(ax.get_xticklabels())+0.2)
                ax.set_ylabel('Count', fontsize=12.5)
        
            else:
                xy_crosstab = pd.crosstab(index=df[y_var], columns=[df[x_var], df[clusters_var]], dropna=False).sort_index(ascending=False, axis=0).reindex(order.get(y_var, df[y_var].unique()) if type(order) == dict else df[y_var].unique(), axis=0) 
                xy_crosstab = xy_crosstab.sort_index(axis=1, level=[1,1]).reindex(order.get(x_var,df[x_var].unique()) if order!= None else df[x_var].unique(), axis=1, level=0)
                colors_dict={'Cluster 1': colors[0], 'Cluster 2': colors[1], 'Cluster 3': colors[2]}
                ax=xy_crosstab.plot(kind='bar', ax=plt.gca(), color=[colors_dict[i] for i in xy_crosstab.columns.get_level_values(1)], alpha=.8 if len(np.unique(df[x_var])) < 15 else 1, edgecolor='lightgray', rot=0)
                ax.set_title(f'Relationship between {title_y} and {title_x}', fontsize=15)
                ax.set_xlabel(y_var, fontsize=12.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if (len(np.unique(df[y_var])) > 4 and y_var != 'Age Group') else None))
                ax.set_xlim(-1,len(ax.get_xticklabels())+0.2)
                ax.set_ylabel('Count', fontsize=12.5)

                #Add hatching pattern to mark empty bars
                for cluster_idx, hatch_value in enumerate(xy_crosstab.columns.get_level_values(0)):
                    for bar in ax.containers[cluster_idx].patches:
                        bar.set_hatch('x') if hatch_value == np.unique(df[x_var])[0] else bar.set_hatch('')  
                
                #Add a horizontal line to indicate empty bar
                for cluster_idx in range(len(ax.containers)):
                    for bar in ax.containers[cluster_idx]:
                        if bar.get_height() == 0:
                            x = bar.get_x() + bar.get_width() / 2
                            y = bar.get_y()  
                            ax.plot([x - bar.get_width() / 2, x + bar.get_width() / 2], [y, y], color=[colors*2][0][cluster_idx], linestyle='-', linewidth=2)

                #adjust the legend's handles 
                legend_handles = []
                for cluster,color in colors_dict.items():
                    for hatch_value in np.unique(df[x_var])[::-1]:
                        legend_handles.append(Patch(facecolor=color, edgecolor='lightgray', hatch='xxx' if hatch_value == np.unique(df[x_var])[0] else '', label=f'{cluster} | {hatch_value}'))
                ax.legend(handles=legend_handles, title=f'Clusters | {x_var}', loc='upper right', alignment='left')

#Define helper function to analyze data and visualize the results
def Get_Plots(x_vars: str | list, y_vars : str | list, clusters_var: str = None, colors: any = plt.get_cmap('tab10'), order: dict = None, **kwargs):
    for y_var in pd.Index(y_vars):
        fig = plt.figure(facecolor='ghostwhite', dpi=150)
        if clusters_var is not None:
            n_cols, n_rows = 4, 4
            fig.set_size_inches(40, 35)
            plt.subplots_adjust(wspace=.28, hspace=.28, top=.94)
            plt.suptitle(f'Customer Segmentation by {y_var}', fontsize=33.5)
        else:
            n_cols = kwargs.get('n_cols', 4 if len(x_vars) - 1 > 5 else len(x_vars))
            n_rows = kwargs.get('n_rows', math.ceil(len(x_vars) / n_cols))
            fig.set_size_inches(12*n_cols, 10*n_rows)
            plt.subplots_adjust(wspace=.28, hspace=.28, top=.92)
            plt.suptitle(f'Bivariate Analysis by {y_var}', fontsize=33.5)

        for i,x_var in enumerate(pd.Index(x_vars).drop(labels=y_var, errors='ignore')):
            #Create subplot for current variables
            plt.subplot(n_rows, n_cols, i+1)
            title_x, title_y = x_var, y_var
            Num_x_categories, Num_y_categories = len(np.unique(df[x_var])), len(np.unique(df[y_var]))

            #Adjust type of plot based on data types of the variables
            if df[x_var].dtype != 'object' and df[y_var].dtype != 'object':
                #visualize data using scatter plot
                color_palette = get_colors(title_y, colors)
                get_scatterplot(x_var, y_var, clusters_var, color_palette)

            elif df[x_var].dtype == 'object' and df[y_var].dtype != 'object':  
                #visualize data using boxen plot
                color_palette = get_colors(title_y, colors)
                boxen = kwargs.get('boxen', True) if len(df[x_var].unique())>2 else True
                get_boxenplot(x_var, y_var, title_x, title_y, clusters_var, boxen, color_palette, order)

            elif df[x_var].dtype != 'object' and df[y_var].dtype == 'object':
                #switch variables on the xy axes
                title_x, title_y = x_var, y_var
                x_var, y_var, Num_x_categories, Num_y_categories = y_var, x_var, Num_y_categories, Num_x_categories
                switched = True 

                if title_x == 'Review Rating':
                    #visualize data using bar plot
                    color_palette = get_colors(title_y, colors)
                    get_barplot(x_var, y_var, title_x, title_y, clusters_var, color_palette, order)
                else:
                    #visualize data using boxen plot
                    color_palette = get_colors(title_y, colors)
                    boxen = kwargs.get('boxen', True) if len(df[x_var].unique())>2 else True
                    get_boxenplot(x_var, y_var, title_x, title_y, clusters_var, boxen, color_palette, order)


            elif df[x_var].dtype == 'object' and df[y_var].dtype == 'object':
                title_x, title_y = x_var, y_var
                if Num_x_categories > Num_y_categories:
                    #switch variables on the xy axes
                    x_var, y_var, Num_x_categories, Num_y_categories = y_var, x_var, Num_y_categories, Num_x_categories
                    switched = True 
                
                heatmap_conditions = ((clusters_var is not None) and ((Num_x_categories > 2) or (Num_x_categories==2 and Num_y_categories>8) or (title_x=='Gender' and len(np.unique(df[title_y]))>2)))
                if heatmap_conditions:
                    #visualize data using heatmap 
                    color_palette = get_colors(title_y, colors)
                    get_heatmap(x_var, y_var, title_x, title_y, clusters_var, color_palette, order)
                else:
                    if kwargs.get('pie', False) and Num_y_categories==4:
                        #visualize data using pie plot 
                        color_palette = get_colors(title_y, colors)
                        get_pieplot(x_var, y_var, title_x, title_y, color_palette)
                    else:
                        #visualize data using bar plot
                        color_palette = get_colors(title_y, colors)
                        get_barplot(x_var, y_var, title_x, title_y, clusters_var, color_palette, order)

                        
            #return variables back to original for next iteration
            try:
                if switched==True:
                    x_var, y_var, Num_x_categories, Num_y_categories = y_var, x_var, Num_y_categories, Num_x_categories
                    switched = False
            except:
                continue 
                
        plt.show()
        if len(y_vars) > 1:
            display(Markdown('<div style="text-align: center;"><br><hr style="border-top: 1.5px solid black; width: 55%;"><br></div>'))



#PART ONE: READING AND INSPECTING THE DATA
#Loading and reading the dataset
#Access and read data into dataframe
df = pd.read_csv('shopping customers dataset.csv').drop('Customer ID',axis=1)

#Preview the first 10 entries
df.head(10)


#Inspecting the Data
#Report the shape of the dataframe
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])


#Checking the data type and number of entries
#Inspect coloumn headers, data type, and number of entries
print(df.info())


#Checking for missing entries 
#Report number of missing values per column
print('Number of missing values per column:')
print(df.isna().sum())


#Checking for data duplicates 
#Report number of duplicates
print('Number of duplicate values: ', df.duplicated().sum())


#Based on present data inspection, it seems that there are no missing or NaN (not a number) entries in the data, no duplicates, and all the data are in the correct 
# data format. Next I will update and enrich the data by creating two new columns to represent age group and region by state.



#PART TWO: UPDATING THE DATA 
#Creating a column for age group 
#Specify age bins 
age_bins = [17, 25, 35, 45, 55, 65, float('inf')]

#Specify labels for the age bins
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']    #i.e., young adults, adults, middle-aged adults, older adults, seniors & elderly

#Perform data binning on the Age column to get new Age Groups column
age_group_col = pd.cut(df['Age'], bins=age_bins, labels=age_labels, ordered=True).astype('object')
df.insert(1, 'Age Group', age_group_col)

#preview obtained age groups
df[['Age', 'Age Group']].sample(10)


#Creating a column for region by state 
#specify the regions by states 
regions_dict = {
    'Far West': ['Alaska', 'California', 'Hawaii', 'Nevada', 'Oregon', 'Washington'],
    'Great Lakes': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin'],
    'Mideast': ['Delaware', 'District of Columbia', 'Maryland', 'New Jersey', 'New York', 'Pennsylvania'],
    'New England': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont'],
    'Plains': ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
    'Rocky Mountains': ['Colorado', 'Idaho', 'Montana', 'Utah', 'Wyoming'],
    'Southeast': ['Alabama', 'Arkansas', 'Florida', 'Georgia', 'Kentucky', 'Louisiana', 'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee', 'Virginia', 'West Virginia'],
    'Southwest': ['Arizona', 'New Mexico', 'Oklahoma', 'Texas'] }

#Create a state to region dictionary
state_to_region = {state: region for region, states in regions_dict.items() for state in states}

#Create new regions column
region_col = df['Location'].map(lambda state: state_to_region.get(state))
df.insert(7, 'Region', region_col)

#preview the obtained regions
df[['Location', 'Region']].sample(10)




#PART THREE: EXPLORATORY DATA ANALYSIS
#Descriptive Statistics 
#Numerical Data
#Get statistical summary of the numerical data
display(df.describe().round(2).T, Markdown('<br>'))

#Show frequency distribution of numerical data using histogram
plt.figure(figsize=(12,9), facecolor='ghostwhite')
plt.suptitle('Frequency Distribution for Numerical Variables', fontsize=14.5)
plt.subplots_adjust(hspace=.25, wspace=.25, top=.94)
for i, col in enumerate(df.select_dtypes(exclude='object')):
    plt.subplot(2,2,i+1)
    ax=sns.histplot(data=df, x=col, bins=10, color='#4C72B0')
    ax.set_xlabel(str(col), fontsize=11)
    ax.set_ylabel('Total Count', fontsize=11)
plt.show()


#Categorical Data
#Get statistical summary of non-numeric (categorical) data 
display(df.describe(include='object').T, Markdown('<br><br>'))

#Show distribution of categorical data
n_rows, n_cols = 3,5
plt.figure(figsize=(46,30), facecolor='ghostwhite', dpi=150)
plt.suptitle('Count Distribution for Categorical Variables', fontsize=40)
plt.subplots_adjust(hspace=.3, top=.94)
for i,col in enumerate(df.select_dtypes(include='object').columns):
    order_dict = {'Age Group': age_labels, 'Size': ['S', 'M', 'L', 'XL'], 'Season': ['Winter', 'Spring', 'Summer', 'Fall'], 'Frequency of Purchases': ['Bi-Weekly', 'Weekly', 'Fortnightly', 'Monthly', 'Every 3 Months', 'Quarterly', 'Annually']}
    plt.subplot(n_rows, n_cols, i+1)
    if len(df[col].unique())==2 or col=='Category':
        ax=plt.gca()
        ax.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.0f%%', labeldistance=1.05, startangle=85, colors=['#355A8D', '#4C72B0', '#6A89C7', '#8CA3D9'], frame=True, wedgeprops=dict(linewidth=1.2, alpha=.83), textprops=dict(fontsize=12))
        ax.patch.set_facecolor((ax.get_facecolor(),0.95))
        ax.set_xlabel(str(col), fontsize=14, labelpad=12)
        ax.set_ylabel('%', fontsize=15, labelpad=12)
        ax.set_xlim(xmin=-1.115, xmax=1.115)
        ax.set_ylim(ymin=-1.1, ymax=1.1)
        ax.set(xticks=[], yticks=[])
    else:
        ax=sns.countplot(x=df[col], color='#4C72B0', order=order_dict.get(col, None))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=(0 if (len(np.unique(df[col])) <= 4 or col=='Age Group') else 60 if (4 < len(np.unique(df[col])) < 10) else 90))
        ax.set_xlabel(str(col), fontsize=14, labelpad=12)
        ax.set_ylabel('Total Count', fontsize=13)
plt.show()

  

#BIVARIATE ANALYSIS 
#Bivariate Analysis by Gender 
#Define target variable and features to compare it to 
target = ['Gender']
features = ['Age', 'Age Group', 'Purchase Amount (USD)', 'Previous Purchases', 'Season', 'Frequency of Purchases', 'Category', 'Region',  'Review Rating', 'Subscription Status']

#Analyze data and plot results 
Get_Plots(features, target, colors=['#3b5998', '#b92b27'], order={'Age Group': age_labels}, pie=True, n_rows=2, n_cols=5)


#Bivariate Analysis by Age Group 
#Define target and relevant features
target = ['Age Group']
features = ['Gender', 'Purchase Amount (USD)', 'Previous Purchases', 'Season', 'Frequency of Purchases', 'Category', 'Region', 'Review Rating', 'Subscription Status']

#Analyze data and plot results 
Get_Plots(features, target, colors='vlag', order={'Age Group': age_labels}, boxen=False, n_rows=2, n_cols=5)


#Bivariate Analysis by Purchase Amount 
#Define target and relevant features
target = ['Purchase Amount (USD)']
features = ['Gender', 'Age', 'Age Group', 'Previous Purchases', 'Season', 'Frequency of Purchases', 'Category', 'Region', 'Review Rating', 'Subscription Status',  'Discount Applied', 'Promo Code Used']

#Analyze data and plot results 
Get_Plots(features, target, colors='vlag', order={'Age Group': age_labels}, boxen=False)


#Bivariate Analysis by Frequency of Purchases 
#Define target and relevant features
target = ['Frequency of Purchases']
features = ['Gender', 'Age', 'Age Group', 'Purchase Amount (USD)', 'Previous Purchases', 'Season', 'Frequency of Purchases', 'Category', 'Region', 'Review Rating', 'Subscription Status',  'Discount Applied', 'Promo Code Used']

#Analyze data and plot results 
Get_Plots(features, target, colors='vlag', order={'Age Group': age_labels}, boxen=False)


#Bivariate Analysis by Region 
#Define target and relevant features
target = ['Region']
features = ['Gender', 'Age', 'Age Group', 'Purchase Amount (USD)', 'Previous Purchases', 'Season', 'Frequency of Purchases', 'Category', 'Review Rating', 'Subscription Status']

#Analyze data and plot results 
Get_Plots(features, target, colors='vlag', order={'Age Group': age_labels}, boxen=False)



#PART FOUR: DATA PREPROCESSING 
#Dealing with Categorical Variables: One-Hot Encoding
#Identify categorical variables
categorical_cols = df.select_dtypes(include='object').columns

#Now we can perform one-hot encoding on the identified columns
#Create encoder object 
OHE_encoder = OneHotEncoder(handle_unknown='ignore')

#Perform One-Hot encoding and return new dataframe with the variables encoded
df_encoded_vars = pd.DataFrame(OHE_encoder.fit_transform(df[categorical_cols]).toarray())
df_encoded_vars.columns = OHE_encoder.get_feature_names_out(categorical_cols)

#Create new dataframe joining the new encoded categories with the earlier numerical variables
df_encoded = pd.concat([df.drop(categorical_cols,axis=1), df_encoded_vars], axis=1)

#Examine dataframe shape after encoding
print('Number of coloumns:', df_encoded.shape[1])
print('Number of rows:', df_encoded.shape[0])
print()

#preview head of the new dataframe 
df_encoded.head()


#Feature Scaling
#Create scaler object
scaler = MinMaxScaler()

#Perform feature normalization 
df_encoded = scaler.fit_transform(df_encoded)

#Now we can look at the value distribution of data after rescaling 
stats_table = pd.DataFrame(df_encoded, columns=scaler.get_feature_names_out()).describe(percentiles=[]).round(1).T
stats_table



#PART FIVE: MODEL DEVELOPMENT AND EVALUATION (CLUSTER ANALYSIS)
#Model Tuning, Evaluation and Comparison
#Define clustering algorithms to test 
estimators_lst = [('K-Means', KMeans(init='k-means++', n_init=10, random_state=42)), 
              ('HAC', AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward', compute_full_tree=True)), 
              ('DBSCAN', DBSCAN(min_samples=50, n_jobs=-1)), 
              ('MS', MeanShift(cluster_all=False, n_jobs=-1))]

#Define parameters to tune for each separate algorithm
bandw = estimate_bandwidth(df_encoded, quantile=.3) 
params_lst = [('n_clusters', np.arange(1,11)),
            ('distance_threshold', np.linspace(20,100,9)),
            ('eps', np.arange(0.1, 1.1, 0.1)),
            ('bandwidth', np.linspace(bandw-1, bandw+1, 10))]

#Create empty table to store tuning results per model
models_results = []

#Loop over each model, optimize and evaluate it and store results
for estimator, params in zip(estimators_lst,params_lst):
    for param in params[1]:
        #Set current parameter value and fit the model
        estimator[1].set_params(**{params[0]: param})
        estimator[1].fit(df_encoded)
        #get model clusters
        clusters = estimator[1].labels_
        
        #compute Silhouette and DBI scores        
        try:
            silhouette = round(silhouette_score(df_encoded, clusters),3)
            davies_bouldin = round(davies_bouldin_score(df_encoded, clusters),3)
        except:
            silhouette, davies_bouldin = np.nan, np.nan
        
        #store model results
        models_results.append({'Model': estimator[0], 'n_clusters': len(np.unique(clusters)), 
                        params[0]: round(param,2), 'silhouette score': silhouette, 'DBI score': davies_bouldin })


#Convert results to dataframe
results_df = pd.DataFrame(models_results).sort_values(['Model','n_clusters']).set_index(keys=['Model', 'n_clusters'])

#report evaluation results by silhouette and DBI score
display(results_df[['silhouette score','DBI score']].drop_duplicates(keep='last'))


#Report and plot evaluation results for each model
fig, axes = plt.subplots(1,2, figsize=(12,6))
model_labels = ['K-Means', 'HAC', 'DBSCAN', 'MS']
for model,params in zip(model_labels,params_lst):
    cols = [params[0], 'silhouette score', 'DBI score']
    cols = [col for col in cols if col != 'n_clusters']
    model_res_df = results_df.iloc[results_df.index.get_level_values(0) == model][cols]
    
    #Report results table per model
    print(f'\nParameter evaluation results for {model} model:')
    display(model_res_df, Markdown('<br>'))
    
    #plot Silhouette scores per model
    ax1=sns.lineplot(data=model_res_df, x=model_res_df.index.get_level_values(1), y='silhouette score', ax=axes[0], label=model)
    sns.scatterplot(data=model_res_df, x=model_res_df.index.get_level_values(1), y='silhouette score', marker='s', ax=ax1)
    ax1.set_title('Number of clusters and Silhouette score',fontsize=12)
    ax1.set(xlabel='Number of clusters', ylabel='Silhouette Score')
    ax1.set_ylim(0, results_df['silhouette score'].max()+.02)
    ax1.legend(loc='upper right', title='Models')
    
    #plot DBI scores per model 
    ax2=sns.lineplot(data=model_res_df, x=model_res_df.index.get_level_values(1), y='DBI score', ax=axes[1], label=model)
    sns.scatterplot(data=model_res_df, x=model_res_df.index.get_level_values(1), y='DBI score', marker='s', ax=ax2)
    ax2.set_title('Number of clusters and Davies Bouldin score', fontsize=12)
    ax2.set(xlabel='Number of clusters', ylabel='Davies Bouldin Score')
    ax2.set_ylim(0, results_df['DBI score'].max()+2)
    ax2.legend(loc='upper right', title='Models')



#FINAL MODEL SELECTION: K-Means Clustering (n_clusters = 3)
#Create k-means object with 3 clusters
Kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

#Fit the K-means model
Kmeans_model = Kmeans.fit(df_encoded)

#Obtain cluster labels and add them to dataframe
df['KM Clusters'] = Kmeans_model.labels_
df['KM Clusters'] = pd.Categorical(df['KM Clusters'].map({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}))



#PART SIX: MODEL INTERPRETAION
#Cluster Analysis: Bivariate Analysis
#Get colors for each cluster 
cluster_colors = plt.get_cmap('Set1_r').colors[-3:]    #green, blue, red

#Define relevant variables 
cols = ['Gender', 'Age', 'Age Group', 'Region', 'Purchase Amount (USD)', 'Frequency of Purchases', 
        'Previous Purchases', 'Season', 'Category', 'Subscription Status', 'Discount Applied', 'Promo Code Used']

#Plot customer segmentation by variable
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(60, 20), facecolor='ghostwhite')
plt.suptitle(f'Customer Segmentation per Variable', fontsize=25)
plt.subplots_adjust(wspace=.2, hspace=.3, top=.91)
for i,col in enumerate(cols):
    ax = axes[i // 6, i % 6]
    if df[col].dtype != 'object':
        sns.boxplot(data=df, x='KM Clusters', y=col, palette=cluster_colors, ax=ax)
        ax.set_title(f'Relationship Between Customer Clusters and {col}', fontsize=14, pad=12)
        ax.set_xlabel('Customer Clusters', fontsize=12, labelpad=5)
        ax.set_xlim(-1,len(ax.get_xticklabels())+0.2) 
        ax.set_ylabel(col, fontsize=12)
    else:  
        sns.countplot(data=df, x=col, hue='KM Clusters', order=order_dict.get(col, None), palette=cluster_colors, gap=.25, ax=ax)
        ax.set_title(f'Relationship Between Customer Clusters and {col}', fontsize=14, pad=12)
        ax.set_xlabel(col, fontsize=12, labelpad=5)
        ax.set_xlim(-1,len(ax.get_xticklabels())+0.2) 
        ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=(0 if (len(ax.get_xticklabels()) <= 4 or col=='Age Group') else 60 if (4 < len(ax.get_xticklabels()) < 10) else 90))
        ax.set_ylabel('Total Count', fontsize=12)
plt.show()



#Cluster Analysis: Multivariate Analysis 
#Get independent variables, x_vars
x_vars = df.columns.drop(labels=['Location', 'Color', 'KM Clusters'])

#Get dependent variables, y_vars
y_vars = ['Gender', 'Age', 'Purchase Amount (USD)', 'Frequency of Purchases', 'Previous Purchases', 'Subscription Status']

#Create order dictionary for better presentation of the variables
order_dict = {'Age Group': age_labels, 
              'Size': ['S', 'M', 'L', 'XL'], 
              'Season': ['Winter', 'Spring', 'Summer', 'Fall'], 
              'Frequency of Purchases': ['Bi-Weekly', 'Weekly', 'Fortnightly', 'Monthly', 'Every 3 Months', 'Quarterly', 'Annually']}


#Perform multivariate analysis and report results 
Get_Plots(x_vars=x_vars, y_vars=y_vars, clusters_var='KM Clusters', colors=cluster_colors, order=order_dict)



#PART SEVEN: INSIGHTS & RECOMMENDATIONS 
# Now as described at length above, clustering analysis yielded three separate clusters, two of which are male dominated and the other is female dominated. What seem to distinguish 
# these three groups most, aside from gender, is loyalty to the brand, purchasing quantities, shopping frequency and subscription to the brand services and related benefits such as 
# discounts and promo codes. One of the male groups is comprised of frequent adult customers, with consistent shopping behavior across the year, a lot of whom are subscribed to the 
# brand and enjoy many benefits in return. The second group of males is comprised of less frequent or seasonal shoppers, mostly shopping in the fall, arguably the shopping season of 
# the year, and whom enjoy no subscriptions or related benefits. Lastly, the last customer group obtained consists of mostly adult female customers whose shopping often tends to be 
# either very frequent (twice a week) or very infrequent (annually). This group also have no subscriptions to the brand and enjoy no benefits. Further analysis has been conducted to 
# examine all the subtle interrelations between all the important variables in the data and in relation to customer group. 
# Here are some key insights and recommendations to increase sales or curate better marketing campaigns based on the analysis results obtained:
#
#  - Generally, the current brand would benefit most from advertising more to younger adults and to female customers, catering to their particular needs and shopping habits, from promoting 
#   subscriptions to their services, and by increasing advertisement efforts and/or opening more branches in the regions with least sales, most notably in the Southwest, Rocky Mountains, 
#   and the Mideast. 
#
# - First off, it's clear from the data that three customer groups do not differ much in their spending capacity. Instead, as consistently illustrated over and again, customer engagement 
#   increases most when customers are subscribed to the brand and are provided certain benefits like discounts and promo codes in return. And thus it seems that the highest driving force 
#   behind sales here is subscription and related benefits. In fact, the presence of discounts and promo codes seem to increase the number of loyal customers and sales coming from this group 
#   without really affecting overall spending. That is to say, such benefits seem to lure customers independent of the benefits they actually bear in absolute monetary terms. Indeed, this is 
#   the case across the three groups as well as within the loyal male customers group: the overall number of loyal customers without subscribtions or benefits is generally lower compared to 
#   the subscribed ones. Discounts and other attractive offers seem to rile people in independent of their spending capabilities, and perhaps also independent of their satisfaction as indicated 
#   by their review rating scores (albeit unsatisfied customers generally do not seem to give any rating, so caution should be taken when interpreting this piece of data). Accordingly, it seems 
#   particularly important to address young adults and female customers, tailoring an advertisement campaign just for these populations and/or offering more discounts and benefits and facilitating 
#   the acquisition of memberships or subscriptions to the service. Subscriptions, discounts and other benefits seem especially important for sales here given that the most loyal group of customers 
#   ushering in the highest amount of sales overall seem to shop exlusively through offers, discounts and promo codes, persumably as a result of their subscription status. Thus, facilitating 
#   subscriptions to the service is highly predictive of increased sales and whatever marketing campaign to be lunched must promote subscriptions and its benefits for the customers especially for 
#   the female group and less frequent male buyers group. 
#
# - Second, as costumers in the Southwest, Rocky Mountains, and the Mideast tend to shop least compared to customers in other regions, the current brand may want to open more branches or increase advertisement 
#   for these regions to ensure better sales. At any rate, each of these regions have their own determinants influencing their sales. One curious aspect about the Southwest in particular, which is associated 
#   with the least sales, is that subscription status doesn't seem to affect sales much in this region. Now as we know subscription status seems to be highly predictive of purchasing behavior, presumably because 
#   of the benefits subscription to the brand confers for the customers like discounts and promo codes. However, it seems that there's little benefit to subscription in the Southwest, as there's little to no 
#   difference between the purchasing behaviors of subscribers and unsubscribers in that region. Thus, efforts have to be spent to ensure that subscriptions actually confer benefits to the customer. 
#
# - Relatedly, as with the Southwest, analysis revealed that subscription status doesn't seem to influence sales much in the Great Lakes, also one the less well performing regions. Now, again, since subscription status 
#   and related benefits seem to be most predictive of sales, the Great Lake branches could improve their sales and benefit most from ensuring that subscribed customers in this region are given reasonable subscription 
#   benefits in return to motivate them to engage more. This is particularly pressing here since looking at the relationship between customer group and purchase amounts in the Great Lakes in particular reveals that 
#   customers generally, and casual male customers especially, tend to spend lower amounts on purchases overall in the Great Lakes branches. So, even when the number of customers is medium or acceptable, they're still 
#   driving lower sales overall. As such, based off all these interacting findings, the Great Lakes branches could greatly improve their sales by, firstly, attending to their male customers better, and secondly, by 
#   improving their subscription services more. 
#
# - Moving to the Mideast in particular, one of the top three regions with least sales, we find that overall sales in this region, as indicated by total purchase amounts, is mostly driven by the loyal male customers group, 
#   but less so by either of the other two customer groups. Now since the Mideast is one of the lowest regions in sales, it seems particularly imperative to address causal male customers and female customers in this region 
#   to bring it to par with the better performing ones. 
#
# - Further, shopping behavior in some of the mentioned regions is also predicted by age. Once again, in the Southwest for instance, young adults in the casual male customers group and young adults in the female customers group 
#   are the least engaged. Thus, more efforts seem to be required in addressing younger adults in these two groups, and, as explicated earlier, in ensuring that subscribed customers actually enjoy good benefits in return, which 
#   would likely ramp up both subscription rates and overall sales in this region. 
#
# - Finally, given that footwear and outerwear are generally sold less relative to other types of clothes, while this might simply reflect there being less items under these two categories for sale, advertisement and marketing efforts 
#   concentrating more on these two categories might increase their sales favorably. This is especially true for female customers as they mostly opt for boots and shoes from this brand. 


