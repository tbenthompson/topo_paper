d1 = 0
d2 = 1
R = np.sqrt(np.sum(proj_m[0] ** 2, axis = 1))
r_vec = np.mean(proj_m[0], axis = 0)

start_vector = r_vec.copy()
start_vector[0] *= 2
v1 = np.cross(r_vec, start_vector)
v1 /= np.linalg.norm(v1)
v2 = np.cross(r_vec, v1)
v2 /= np.linalg.norm(v2)

offset = proj_m[0] - r_vec
X = -offset.dot(v2)
Y = offset.dot(v1)

fault_offset = fault_m[0] - r_vec
fX = -fault_offset.dot(v2)
fY = fault_offset.dot(v1)

plt.figure(figsize = (14,14))
plt.tricontourf(X, Y, proj_m[1], R)
plt.triplot(fX, fY, fault_m[1])
plt.colorbar()
plt.show()