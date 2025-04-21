import csv
from datetime import datetime, timedelta

# Define the date range
start_date = datetime(2013, 4, 28)  # Halfway through 2013 (July 1)
end_date = datetime(2013, 7, 1)    # Halfway through 2015 (July 1)

# Generate the list of dates
date_list = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days)]

# Write to CSV file
csv_filename = "dates_early_2013.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Date"])  # Write header
    writer.writerows([[date] for date in date_list])  # Write dates

print(f"CSV file '{csv_filename}' has been created with {len(date_list)} dates.")

