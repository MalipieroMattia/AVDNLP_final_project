import kagglehub
from kagglehub import KaggleDatasetAdapter

# set the path
file_path = ""

# load the latest version
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "",
    file_path,
)

print("First 5 records:", df.head())
