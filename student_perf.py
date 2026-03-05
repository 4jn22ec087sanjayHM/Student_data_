import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
data = pd.read_csv("student_perf.csv")

print("Dataset Preview:")
print(data.head())

# 2. Dataset Information
print("\nDataset Info:")
print(data.info())

# 3. Statistical Summary
print("\nStatistical Summary:")
print(data.describe())

# 4. Create New Columns (Feature Engineering)
data["Total"] = data["Math"] + data["Science"] + data["English"]
data["Average"] = data["Total"] / 3

print("\nDataset with Total and Average:")
print(data)

# 5. Find Top Student
top_student = data.loc[data["Total"].idxmax()]

print("\nTop Performing Student:")
print(top_student)

# 6. Subject-wise Average
print("\nSubject Averages:")
print(data[["Math","Science","English"]].mean())

# 7. Correlation Analysis
print("\nCorrelation Matrix:")
print(data.corr())

# 8. Scatter Plot (Study Hours vs Average Marks)
plt.figure()
plt.scatter(data["StudyHours"], data["Average"])
plt.xlabel("Study Hours")
plt.ylabel("Average Marks")
plt.title("Study Hours vs Average Marks")
plt.show()

# 9. Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 10. Bar Chart of Subject Averages
subject_avg = data[["Math","Science","English"]].mean()

plt.figure()
subject_avg.plot(kind="bar")
plt.title("Average Marks per Subject")
plt.ylabel("Marks")
plt.show()