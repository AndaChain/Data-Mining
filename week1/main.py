import linear_sin, linear_x2, linear_sin_org, linear_x2_org, constant_sin, constant_x2
import matplotlib.pyplot as plt
N = 10000 # จำนวนรอบ
start = -1 # x น้อยสุด
stop = 1 # x มากสุด
linear_sin.real_func(start, stop, N)
linear_sin.Error(start, stop, N)
linear_sin.set_graph() # set graph
plt.show()
print("-----------------------")

linear_x2.real_func(start, stop, N)
linear_x2.make_model(start, stop, N)
linear_x2.set_graph() # set graph
plt.show()
print("-----------------------")

linear_sin_org.real_func(start, stop, N)
linear_sin_org.make_model(start, stop, N)
linear_sin_org.set_graph() # set graph
plt.show()
print("-----------------------")

linear_x2_org.real_func(start, stop, N)
linear_x2_org.make_model(start, stop, N)
linear_x2_org.set_graph() # set graph
plt.show()
print("-----------------------")

constant_sin.real_func(start, stop, N)
constant_sin.make_model(start, stop, N)
constant_sin.set_graph() # set graph
plt.show()
print("-----------------------")

constant_x2.real_func(start, stop, N)
constant_x2.make_model(start, stop, N)
constant_x2.set_graph() # set graph
plt.show()

