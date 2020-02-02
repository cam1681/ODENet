import sys
import subprocess

print("Start generating the equaiton data!")
subprocess.call(["python", "LV.py"])
print("Start learning the matrix!\n")
subprocess.call(["python", "ode_read_file.py","--viz"])
print("Reconstruct the equation!\n")
subprocess.call(["python", "Reconstruct.py"])
