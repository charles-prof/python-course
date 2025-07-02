import numpy as np

# def main():
#     print("Hello, world!")

# if __name__ == "__main__":
#     main()
def row_with_max_ones(arr: np.ndarray) -> int:
    # Count 1s in each row using np.sum (since all elements are 0 or 1)
    one_counts = np.sum(arr, axis=1)
    
    # Get index of the maximum count
    return int(np.argmax(one_counts))

arr = np.array([
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 1]
])

print(f"The row with the maximum number of 1s is: {row_with_max_ones(arr)}")  # Output: 1
    
