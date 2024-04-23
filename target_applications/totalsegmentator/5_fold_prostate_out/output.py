import os 
import pandas as pd 
dir="/home/azureuser/SuPreM/target_applications/totalsegmentator/5_fold_prostate_out"
dirs = os.listdir(dir)
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
        dice.to_csv(f"{model}_dice.csv")
        nsd.to_csv(f"{model}_nsd.csv")
        