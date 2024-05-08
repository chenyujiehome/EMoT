import os 
import pandas as pd 
dir="/home/fanlinghuang/TAD-chenyujie/SuPreM/target_applications/totalsegmentator/5_fold_prostate_out"
dirs = os.listdir(dir)
dices=[]
nsds=[]
for d in dirs:
    model=d.split(".")[1]
    if os.path.isdir(os.path.join(dir,d)):
        print(d)
        files = os.listdir(os.path.join(dir,d))
        dice=pd.DataFrame()
        nsd=pd.DataFrame()
        for f in files:
            tmp=pd.read_csv(os.path.join(dir,d,f))
            if 'dice' in f:
                dice=pd.concat([dice,tmp],axis=0)
            else:
                nsd=pd.concat([nsd,tmp],axis=0)
        dice = dice.dropna()
        nsd = nsd.dropna()
        # Calculate mean and standard deviation, ignoring non-numeric columns
        # Calculate mean and standard deviation, ignoring non-numeric columns
        dice_mean = dice.mean(numeric_only=True)
        dice_std = dice.std(numeric_only=True)

        # Append mean and standard deviation to the DataFrame
        dice = pd.concat([dice, dice_mean.to_frame().T, dice_std.to_frame().T])

        # Calculate mean and standard deviation, ignoring non-numeric columns
        nsd_mean = nsd.mean(numeric_only=True)
        nsd_std = nsd.std(numeric_only=True)

        # Append mean and standard deviation to the DataFrame
        nsd = pd.concat([nsd, nsd_mean.to_frame().T, nsd_std.to_frame().T])
        dices.append([dice.reset_index(),model])
        nsds.append([nsd.reset_index(),model])
results_dice = []

for dice, model in dices:
    # Select the last two rows and the specified columns
    selected_rows = dice.loc[dice.index[-2:], ['background', 'PZ', 'TZ']]
    
    # Calculate mean and standard deviation
    mean = selected_rows.loc[dice.index[-2]]
    std = selected_rows.loc[dice.index[-1]]
    
    # Format the results as "mean +- std"
    formatted_results = mean.astype(str) + " ± " + std.astype(str)
    
    # Add the model name
    formatted_results['name'] = model
    
    # Append the results to the list
    results_dice.append(formatted_results)

# Convert the list of results to a DataFrame
results_df_dice = pd.DataFrame(results_dice)


results_nsd = []

for nsd, model in nsds:
    # Select the last two rows and the specified columns
    selected_rows = nsd.loc[nsd.index[-2:], ['background', 'PZ', 'TZ']]
    
    # Calculate mean and standard deviation
    mean = selected_rows.loc[nsd.index[-2]]
    std = selected_rows.loc[nsd.index[-1]]
    
    # Format the results as "mean +- std"
    formatted_results = mean.astype(str) + " ± " + std.astype(str)
    
    # Add the model name
    formatted_results['name'] = model
    
    # Append the results to the list
    results_nsd.append(formatted_results)

# Convert the list of results to a DataFrame
results_df_nsd = pd.DataFrame(results_nsd)


# Drop the 'background' column and move the 'name' column to the first position
results_df_dice = results_df_dice.drop(columns='background')
results_df_dice = results_df_dice.reindex(['name'] + list(results_df_dice.columns[:-1]), axis=1)

# Save the results to a CSV file
results_df_dice.to_csv('dice.csv', index=False)

# Drop the 'background' column and move the 'name' column to the first position
results_df_nsd = results_df_nsd.drop(columns='background')
results_df_nsd = results_df_nsd.reindex(['name'] + list(results_df_nsd.columns[:-1]), axis=1)

# Save the results to a CSV file
results_df_nsd.to_csv('nsd.csv', index=False)