import json
import numpy as np

with open('./matrix0/matrix2910.json') as json_file:
    data = json.load(json_file)

vc = np.array(data['vc'])
v = vc[:,0]

k1 = v[5]
c = v[2]/(-2*k1)
a = k1*c**2
k2 = v[4]
k3 = v[1]+k2*c


print('k1, k2, k3, c, a, aor')
print([k1, k2, k3, c, a, v[0]])

