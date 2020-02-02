""" Reconstruct the lv equation through matrix obatined. Created by pi 2019/05/29"""

from sympy import *
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser('Reconstruction')
parser.add_argument('--itr', type=int, default=18000)
args = parser.parse_args()

with open('matrix/matrix{}.json'.format(args.itr)) as json_file:
    data = json.load(json_file)

ma = Matrix(data["ma"])
mb = Matrix(data["mb"])
mc = Matrix(data["mc"])

x, y = symbols('x y')
xx =  Matrix([x,y]).reshape(1,2)

result = expand(matrix_multiply_elementwise(xx*ma,xx*mb) + xx*mc)

print(result)
