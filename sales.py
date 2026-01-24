import pandas as pd

# Data as dictionary
sales_data = {
    "sales": [1200, 1500, 1800, 1100, 2000, 1700, 1600, 1900, 1400, 2100],
    "monthly": [
        "January", "February", "March", "April", "May",
        "June", "July", "August", "September", "October"
    ],
    "product_name": [
        "Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch",
        "Camera", "Printer", "Monitor", "Keyboard", "Mouse"
    ],
    "profit": [300, 450, 380, 220, 520, 410, 360, 480, 200, 550]
}


# Create DataFrame
df = pd.DataFrame(sales_data)

# Save to CSV
df.to_csv("sales.csv", index=False)

print("✅ sales.csv file created successfully!")
