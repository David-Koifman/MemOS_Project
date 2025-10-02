
import pandas as pd

# Load Excel
df = pd.read_excel("results_detailed.xlsx")

# Convert to plain text table (nice alignment)
text_output = df.to_string(index=False)

# Save to txt
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text_output)

print("Table saved to output.txt")