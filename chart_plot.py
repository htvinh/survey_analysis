import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from matplotlib.axes import Axes


# Set constant

option = ['Fintech', 'SIM']
select = 1 # Or 1
if select == 1:
    filename = 'INFO_International_Student_Data.xlsx'
    sheet_name = 'Afrique' # All, Afrique
    image_filename = 'INFO_International_Student.png'
    image_title = 'Informatique Master Afrique Students - IFI Institut - VNU'
    x_label_title, y_label_title = "Nombre d'étudiants", 'Pays' 
    frequence_col = "Nombre d'étudiants"
    organization_col = 'Pays' # All, Afrique
    adjust = 5
elif select == 0:
    filename = 'Fintech_profile_22_SEPT_2023.xlsx'
    sheet_name = 'Summary'
    image_filename = 'Fintech_Master_IFI_Profile.png'
    image_title = 'FINTECH Master Student Profile - IFI Institute - VNU, 2021-2023'
    x_label_title, y_label_title = 'Frequency', 'Organization' 
    frequence_col = "Frequency"
    organization_col = 'Organization'
    adjust = 1

# Read the "Overall Summary" sheet from the Excel file into a DataFrame
file_path = f'./data/{filename}'  # Replace with the actual file path
df = pd.read_excel(file_path, sheet_name=sheet_name)

# df = df.dropna()
# df[frequence_col] = df[frequence_col].astype(int)

# Sort the DataFrame by Frequency for better visualization
sorted_df = df.sort_values(by=[frequence_col, organization_col], ascending=[False, True])

total_frequency = df[frequence_col].sum()
max_frequency = df[frequence_col].max()

# Plotting code starts here
fig, ax = plt.subplots(figsize=(14, 10))

# Plot the bar chart
# Set the maximum value for the x-axis
plt.xlim(0, max_frequency + adjust)  # Adjust the maximum value as needed
ax = sns.barplot(x=x_label_title, y=y_label_title, data=sorted_df, palette='viridis', ax=ax)

# Remove the axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Add titles and labels
# Add titles and labels with bold and blue color for the title
ax.set_title(image_title, fontsize=18, x=0.5, y=1.05, ha='center', weight='bold', color='blue')
# plt.suptitle('Overall Summary of Nơi làm việc', fontsize=16, x=0.5, y=0.98, ha='center')
# plt.xlabel(x_label_title, fontsize=14)
# plt.ylabel(y_label_title, fontsize=14)

# Annotate the bars
for index, value in enumerate(sorted_df[frequence_col]):
    percent = value/total_frequency*100
    ax.text(value, index, f' {value:.0f} ({percent:.1f}%)', va='center', fontsize=10)


# Save the chart
plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
plt.savefig(image_filename, dpi=300)  # Replace with the desired save path
