import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tectosaur.geometry import projection, tri_normal, tri_area

def plot_situation():
    w = np.max(hill_pts[:,0])
    xi = np.linspace(np.min(hill_pts[:,0]),np.max(hill_pts[:,0]),100)
    yi = np.linspace(np.max(hill_pts[:,1]),np.max(hill_pts[:,1]),100)

    z = hill_pts[:, 2]
    zi = griddata((hill_pts[:,0], hill_pts[:,1]), z, (xi[None,:], yi[:,None]), method='cubic')

    min_z, max_z = np.min(z), np.max(z)
    levels = np.linspace(min_z, max_z, 15)[1:]
    CS = plt.contour(xi,yi,zi,levels = levels, linewidths=0.5, colors='k')

    plt.plot(fault[0][:,0], fault[0][:,1], 'r*')
    plt.plot(flat_pts[obs_pt_idxs,0], flat_pts[obs_pt_idxs,1], 'o')
    plt.show()
# plot_situation()

def get_dip_strike_dirs(tri):
    vertical = np.array([0,0,1])
    normal = tri_normal(tri, normalize = True)
    dip_dir = vertical - projection(vertical, normal)
    dip_dir /= np.linalg.norm(dip_dir)
    strike_dir = np.cross(normal, dip_dir)
    return dip_dir, strike_dir

def get_slip_vec_mags(fault, slip_vecs, dip_mag, strike_mag):
    slip_vec_mag = np.empty((fault[1].shape[0] * 2))

    vertical = np.array([0,0,1])
    for i in range(fault[1].shape[0] * 2):
        tri_idx = i // 2
        dip_dir, strike_dir = get_dip_strike_dirs(fault[0][fault[1][tri_idx,:]])
        s = slip_vecs[i,0,:]
        slip_vec_mag[i] = (dip_mag * dip_dir).dot(s) + (strike_mag * strike_dir).dot(s)

    return slip_vec_mag

def surf_disp_from_gfs(gfs, fault, slip_vecs, dip_mag, strike_mag):
    return np.sum(
        get_slip_vec_mags(fault, slip_vecs, dip_mag, strike_mag)[:,np.newaxis, np.newaxis] *
            gfs,
        axis = 0
    )

def forward_model():
    flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/wenchuan/narrow/flat_wenchuan_gfs.npy')
    hill_pts, _, hill_gfs, hill_surf, _ = np.load('data/wenchuan/narrow/hill_wenchuan_gfs.npy')

    surf_disp = surf_disp_from_gfs(hill_gfs, fault, slip_vecs, -1.0, 0.0)
    filename = 'data/wenchuan_disp2.npy'
    np.save(filename, (hill_pts, surf_disp))

    surf_disp = surf_disp_from_gfs(flat_gfs, fault, slip_vecs, -1.0, 0.0)
    filename = 'data/wenchuan_disp2_flat.npy'
    np.save(filename, (flat_pts, surf_disp))

def total_moment(slip, fault):
    M = 0
    for i in range(fault[1].shape[0]):
        A = tri_area(fault[0][fault[1][i,:]])
        s = np.sqrt(slip[i * 2] ** 2 + slip[i * 2 + 1] ** 2)
        M += A * s
    return M

def main():

    np.random.seed(1)

    flat_pts, slip_vecs, flat_gfs, flat_surf, fault = np.load('data/wenchuan/narrow/flat_wenchuan_gfs.npy')
    hill_pts, _, hill_gfs, hill_surf, _ = np.load('data/wenchuan/narrow/hill_wenchuan_gfs.npy')

    # choose some random obs pts?
    # obs_pt_idxs = np.random.randint(0, flat_pts.shape[0], size = (250,))
    # or all obs pts?
    obs_pt_idxs = np.arange(flat_pts.shape[0])

    # use only ux, uy, one row for each observation point.
    which_dims = [0,1]

    # TODO: Set these values.
    dip_slip_mag = 1.0
    strike_slip_mag = 0.0

    # G = hill_gfs
    G = flat_gfs
    correct_x = get_slip_vec_mags(fault, slip_vecs, dip_slip_mag, strike_slip_mag)
    print("correct moment: " + str(total_moment(correct_x, fault)))
    b = surf_disp_from_gfs(
        hill_gfs, fault, slip_vecs, dip_slip_mag, strike_slip_mag
    )[obs_pt_idxs,:][:,which_dims].flatten()

    reg_param = 0.025
    xs = []
    # for reg_param in (2.0 ** -np.arange(0, 15)):
    for G in [flat_gfs, hill_gfs]:
        A = G[:,obs_pt_idxs][:,:,which_dims].reshape((G.shape[0], -1)).T
        A += reg_param * np.eye(*A.shape)
        x = np.linalg.lstsq(A, b)[0]
        xs.append(x)

    # plot_me = np.sqrt(x[1::2] ** 2 + x[::2] ** 2)
    # plt.figure()
    # plt.tripcolor(fault[0][:,0], fault[0][:,2], fault[1], plot_me, shading = 'flat')
    # plt.colorbar()
    # plt.show()

    # plt.savefig(str(reg_param))
    # print(reg_param, x)

    print(reg_param, total_moment(xs[0], fault))
    print(reg_param, total_moment(xs[1], fault))
    np.save('data/wenchuan_inversion_' + str(reg_param) + '.npy', xs)

if __name__ == '__main__':
    main()
