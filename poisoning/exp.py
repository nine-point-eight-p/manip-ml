import subprocess

args = [
    "-s", "0",
    "-p", "64",
    "-r", "467",
    "-t", "467",
    "-v", "467",
    "-m", "lasso"
]

N = 1
results = []

for i in range(N):
    subprocess.run(["python", "poison.py"] + args)
    with open("../results/lasso0/err.txt", "r") as f:
        final = f.readlines()[-1].split(",")
    results.append(final[-3:])

val_mse, test_mse, time = zip(*results)
print("Average validation MSE:", sum(val_mse) / N)
print("Average test MSE:", sum(test_mse) / N)
print("Average time:", sum(time) / N)
