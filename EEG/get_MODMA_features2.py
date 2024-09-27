import pandas as pd

de = pd.read_excel('de_model_outputs.xlsx')
psd = pd.read_excel('psd_model_outputs.xlsx')

mean_df = pd.DataFrame()


for i in range(12):  
    de_column = f'DE_{i}.pth'
    psd_column = f'PSD_{i}.pth'
    
    if de_column in de.columns and psd_column in psd.columns:
        combined_values = de[de_column] + psd[psd_column]
        mean_value = combined_values.mean()
        mean_df[i] = combined_values
        mean_df['filename'] = de['filename'].str.replace('.mat', '')

mean_df.to_excel('mean_values.xlsx', index=False)
mean_df