import jax.ops

from .. import veros_method, runtime_settings as rs, runtime_state as rst
from . import density, diffusion, utilities


@veros_method(dist_safe=False, local_variables=(
    'dxt', 'dxu', 'xt', 'xu',
    'dyt', 'dyu', 'yt', 'yu',
    'dzt', 'dzw', 'zt', 'zw',
    'cost', 'cosu', 'tantr',
    'area_t', 'area_u', 'area_v',
))
def calc_grid(vs):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """

    def u_centered_grid(dyt, dyu, yt, yu):
        yu = jax.ops.index_update(yu, jax.ops.index[0], 0)
        yu = jax.ops.index_update(yu, jax.ops.index[1:], np.cumsum(dyt[1:]))

        yt = jax.ops.index_update(yt, jax.ops.index[0], yu[0] - dyt[0] * 0.5)
        yt = jax.ops.index_update(yt, jax.ops.index[1:], 2 * yu[:-1])

        alternating_pattern = np.ones_like(yt)
        alternating_pattern = jax.ops.index_update(alternating_pattern, jax.ops.index[::2], -1)
        yt = jax.ops.index_update(yt, jax.ops.index[...], alternating_pattern * np.cumsum(alternating_pattern * yt))

        dyu = jax.ops.index_update(dyu, jax.ops.index[:-1], yt[1:] - yt[:-1])
        dyu = jax.ops.index_update(dyu, jax.ops.index[-1], 2 * dyt[-1] - dyu[-2])

    if vs.enable_cyclic_x:
        vs.dxt = jax.ops.index_update(vs.dxt, jax.ops.index[-2:], vs.dxt[2:4])
        vs.dxt = jax.ops.index_update(vs.dxt, jax.ops.index[:2], vs.dxt[-4:-2])
    else:
        vs.dxt = jax.ops.index_update(vs.dxt, jax.ops.index[-2:], vs.dxt[-3])
        vs.dxt = jax.ops.index_update(vs.dxt, jax.ops.index[:2], vs.dxt[2])

    vs.dyt = jax.ops.index_update(vs.dyt, jax.ops.index[-2:], vs.dyt[-3])
    vs.dyt = jax.ops.index_update(vs.dyt, jax.ops.index[:2], vs.dyt[2])

    """
    grid in east/west direction
    """
    u_centered_grid(vs.dxt, vs.dxu, vs.xt, vs.xu)
    vs.xt += vs.x_origin - vs.xu[2]
    vs.xu += vs.x_origin - vs.xu[2]

    if vs.enable_cyclic_x:
        vs.xt = jax.ops.index_update(vs.xt, jax.ops.index[-2:], vs.xt[2:4])
        vs.xt = jax.ops.index_update(vs.xt, jax.ops.index[:2], vs.xt[-4:-2])
        vs.xu = jax.ops.index_update(vs.xu, jax.ops.index[-2:], vs.xt[2:4])
        vs.xu = jax.ops.index_update(vs.xu, jax.ops.index[:2], vs.xu[-4:-2])
        vs.dxu = jax.ops.index_update(vs.dxu, jax.ops.index[-2:], vs.dxu[2:4])
        vs.dxu = jax.ops.index_update(vs.dxu, jax.ops.index[:2], vs.dxu[-4:-2])

    """
    grid in north/south direction
    """
    u_centered_grid(vs.dyt, vs.dyu, vs.yt, vs.yu)
    vs.yt += vs.y_origin - vs.yu[2]
    vs.yu += vs.y_origin - vs.yu[2]

    if vs.coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        vs.dxt *= vs.degtom
        vs.dxu *= vs.degtom
        vs.dyt *= vs.degtom
        vs.dyu *= vs.degtom

    """
    grid in vertical direction
    """
    u_centered_grid(vs.dzt, vs.dzw, vs.zt, vs.zw)
    vs.zt -= vs.zw[-1]
    vs.zw -= vs.zw[-1]  # enforce 0 boundary height

    """
    metric factors
    """
    if vs.coord_degree:
        vs.cost = jax.ops.index_update(vs.cost, jax.ops.index[...], np.cos(vs.yt * vs.pi / 180.))
        vs.cosu = jax.ops.index_update(vs.cosu, jax.ops.index[...], np.cos(vs.yu * vs.pi / 180.))
        vs.tantr = jax.ops.index_update(vs.tantr, jax.ops.index[...], np.tan(vs.yt * vs.pi / 180.) / vs.radius)
    else:
        vs.cost = jax.ops.index_update(vs.cost, jax.ops.index[...], 1.0)
        vs.cosu = jax.ops.index_update(vs.cosu, jax.ops.index[...], 1.0)
        vs.tantr = jax.ops.index_update(vs.tantr, jax.ops.index[...], 0.0)

    """
    precalculate area of boxes
    """
    vs.area_t = jax.ops.index_update(vs.area_t, jax.ops.index[...], vs.cost * vs.dyt * vs.dxt[:, np.newaxis])
    vs.area_u = jax.ops.index_update(vs.area_u, jax.ops.index[...], vs.cost * vs.dyt * vs.dxu[:, np.newaxis])
    vs.area_v = jax.ops.index_update(vs.area_v, jax.ops.index[...], vs.cosu * vs.dyu * vs.dxt[:, np.newaxis])


@veros_method
def calc_beta(vs):
    """
    calculate beta = df/dy
    """
    vs.beta = jax.ops.index_update(vs.beta, jax.ops.index[:, 2:-2],
        0.5 * ((vs.coriolis_t[:, 3:-1] - vs.coriolis_t[:, 2:-2]) / vs.dyu[2:-2]
                            + (vs.coriolis_t[:, 2:-2] - vs.coriolis_t[:, 1:-3]) / vs.dyu[1:-3])
    )

    utilities.enforce_boundaries(vs, vs.beta)


@veros_method
def calc_topo(vs):
    """
    calulate masks, total depth etc
    """

    """
    close domain
    """

    vs.kbot = jax.ops.index_update(vs.kbot, jax.ops.index[:, :2], 0)
    vs.kbot = jax.ops.index_update(vs.kbot, jax.ops.index[:, -2:], 0)

    utilities.enforce_boundaries(vs, vs.kbot)

    if not vs.enable_cyclic_x:
        vs.kbot = jax.ops.index_update(vs.kbot, jax.ops.index[:2, :], 0)
        vs.kbot = jax.ops.index_update(vs.kbot, jax.ops.index[-2:, :], 0)

    """
    Land masks
    """
    vs.maskT = jax.ops.index_update(vs.maskT, jax.ops.index[...], 0.0)
    land_mask = vs.kbot > 0
    ks = np.arange(vs.maskT.shape[2])[np.newaxis, np.newaxis, :]
    vs.maskT = jax.ops.index_update(vs.maskT, jax.ops.index[...],
        land_mask[..., np.newaxis] & (vs.kbot[..., np.newaxis] - 1 <= ks)
    )
    utilities.enforce_boundaries(vs, vs.maskT)
    vs.maskU = jax.ops.index_update(vs.maskU, jax.ops.index[...],
        vs.maskT)
    vs.maskU = jax.ops.index_update(vs.maskU, jax.ops.index[:-1, :, :],
        np.minimum(vs.maskT[:-1, :, :], vs.maskT[1:, :, :]))
    utilities.enforce_boundaries(vs, vs.maskU)
    vs.maskV = jax.ops.index_update(vs.maskV, jax.ops.index[...],
        vs.maskT)
    vs.maskV = jax.ops.index_update(vs.maskV, jax.ops.index[:, :-1],
        np.minimum(vs.maskT[:, :-1], vs.maskT[:, 1:]))
    utilities.enforce_boundaries(vs, vs.maskV)
    vs.maskZ = jax.ops.index_update(vs.maskZ, jax.ops.index[...],
        vs.maskT)
    vs.maskZ = jax.ops.index_update(vs.maskZ, jax.ops.index[:-1, :-1],
        np.minimum(np.minimum(vs.maskT[:-1, :-1],
                                                      vs.maskT[:-1, 1:]),
                                                 vs.maskT[1:, :-1])
    )
    utilities.enforce_boundaries(vs, vs.maskZ)
    vs.maskW = jax.ops.index_update(vs.maskW, jax.ops.index[...],
        vs.maskT)
    vs.maskW = jax.ops.index_update(vs.maskW, jax.ops.index[:, :, :-1],
        np.minimum(vs.maskT[:, :, :-1], vs.maskT[:, :, 1:]))

    """
    total depth
    """
    vs.ht = jax.ops.index_update(vs.ht, jax.ops.index[...],
        np.sum(vs.maskT * vs.dzt[np.newaxis, np.newaxis, :], axis=2))
    vs.hu = jax.ops.index_update(vs.hu, jax.ops.index[...],
        np.sum(vs.maskU * vs.dzt[np.newaxis, np.newaxis, :], axis=2))
    vs.hv = jax.ops.index_update(vs.hv, jax.ops.index[...],
        np.sum(vs.maskV * vs.dzt[np.newaxis, np.newaxis, :], axis=2))

    mask = (vs.hu == 0).astype('float')
    vs.hur = jax.ops.index_update(vs.hur, jax.ops.index[...],
        1. / (vs.hu + mask) * (1 - mask))
    mask = (vs.hv == 0).astype('float')
    vs.hvr = jax.ops.index_update(vs.hvr, jax.ops.index[...],
        1. / (vs.hv + mask) * (1 - mask))


@veros_method
def calc_initial_conditions(vs):
    """
    calculate dyn. enthalp, etc
    """
    if np.any(vs.salt < 0.0):
        raise RuntimeError('encountered negative salinity')

    utilities.enforce_boundaries(vs, vs.temp)
    utilities.enforce_boundaries(vs, vs.salt)

    vs.rho = jax.ops.index_update(vs.rho, jax.ops.index[...],
        density.get_rho(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]) \
                  * vs.maskT[..., np.newaxis])
    vs.Hd = jax.ops.index_update(vs.Hd, jax.ops.index[...],
        density.get_dyn_enthalpy(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]) \
                 * vs.maskT[..., np.newaxis])
    vs.int_drhodT = jax.ops.index_update(vs.int_drhodT, jax.ops.index[...],
        density.get_int_drhodT(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]))
    vs.int_drhodS = jax.ops.index_update(vs.int_drhodS, jax.ops.index[...],
        density.get_int_drhodS(vs, vs.salt, vs.temp, np.abs(vs.zt)[:, np.newaxis]))

    fxa = -vs.grav / vs.rho_0 / vs.dzw[np.newaxis, np.newaxis, :] * vs.maskW
    vs.Nsqr = jax.ops.index_update(vs.Nsqr, jax.ops.index[:, :, :-1, :],
        fxa[:, :, :-1, np.newaxis] \
        * (density.get_rho(vs, vs.salt[:, :, 1:, :], vs.temp[:, :, 1:, :], np.abs(vs.zt)[:-1, np.newaxis])
         - vs.rho[:, :, :-1, :]))
    vs.Nsqr = jax.ops.index_update(vs.Nsqr, jax.ops.index[:, :, -1, :],
        vs.Nsqr[:, :, -2, :])


@veros_method(inline=True)
def ugrid_to_tgrid(vs, a):
    b = np.zeros_like(a)
    b = jax.ops.index_update(b, jax.ops.index[2:-2, :, :],
        (vs.dxu[2:-2, np.newaxis, np.newaxis] * a[2:-2, :, :] + vs.dxu[1:-3, np.newaxis, np.newaxis] * a[1:-3, :, :]) \
        / (2 * vs.dxt[2:-2, np.newaxis, np.newaxis])
    )
    return b


@veros_method(inline=True)
def vgrid_to_tgrid(vs, a):
    b = np.zeros_like(a)
    b = jax.ops.index_update(b, jax.ops.index[:, 2:-2, :],
        (vs.area_v[:, 2:-2, np.newaxis] * a[:, 2:-2, :] + vs.area_v[:, 1:-3, np.newaxis] * a[:, 1:-3, :]) \
        / (2 * vs.area_t[:, 2:-2, np.newaxis])
    )
    return b


@veros_method
def solve_tridiag(vs, a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = np.stack((cp, dp))
        return new_primes, new_primes

    diags_stacked = np.stack(
        [arr.transpose((2, 0, 1)) for arr in (a, b, c, d)],
        axis=1
    )
    _, primes = jax.lax.scan(compute_primes, np.zeros((2, *a.shape[:-1])), diags_stacked)

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, np.zeros(a.shape[:-1]), primes[::-1])
    return sol[::-1].transpose((1, 2, 0))


@veros_method(inline=True)
def calc_diss(vs, diss, tag):
    diss_u = np.zeros_like(diss)
    ks = np.zeros_like(vs.kbot)
    if tag == 'U':
        ks = jax.ops.index_update(ks, jax.ops.index[1:-2, 2:-2],
            np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1)
        interpolator = ugrid_to_tgrid
    elif tag == 'V':
        ks = jax.ops.index_update(ks, jax.ops.index[2:-2, 1:-2],
            np.maximum(vs.kbot[2:-2, 1:-2], vs.kbot[2:-2, 2:-1]) - 1)
        interpolator = vgrid_to_tgrid
    else:
        raise ValueError('unknown tag {} (must be \'U\' or \'V\')'.format(tag))
    diffusion.dissipation_on_wgrid(vs, diss_u, aloc=diss, ks=ks)
    return interpolator(vs, diss_u)
