import simplejson as json
from numpy.random import default_rng, SeedSequence
import os
import time
import sys

args = sys.argv[1:]  # 从第二个元素开始取，因为第一个元素是程序名称本身

shell_no = args[0]
print("no = ", shell_no)

now = time.time()
print("start_time = ", now)


seed_sequence = SeedSequence()
rng = default_rng(seed_sequence)

sleep_time = int(rng.integers(10, 30))
print("sleep_time = {}".format(sleep_time))
time.sleep(sleep_time)

tx_a = rng.integers(5, 20, (2, 3))
tx_b = rng.integers(10, 20, (3, 4))

tp_a = (10, 20)
tp_b = (1, 2, None)

list1 = [[1, 23, 6, 4, 56], [2,1,2,3,5,4]]

print(tx_a)
print(tx_b)

fig = {
    "list1": list1,
    "tx_a": tx_a.tolist(),
    "tx_b": tx_b.tolist(),
    "tp_a": tp_a,
    "tp_b": tp_b
}

solution = {
    "service_a": [(0, 1), (1, 2)],
    "service_q": (0, 2),
    "service_r": [(0, 0, 1), (0, 1, 2), (1, 0, 4)]
}

result = {
    "fig": fig,
    "sol": solution
}

print(result)

num = int(rng.integers(10, 20))
print("{}, type = {}".format(num, type(num)))

base = 50
step = 40
num = base
for i in range(10):
    print(num)
    num += step

print("end_time = ", time.time())

# filename = "./test.json"
# with open(filename, "w") as file:
#     json.dump(result, file)
