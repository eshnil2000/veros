import jax.ops

from ... import veros_method
from ...variables import allocate
from .. import numerics, utilities


@veros_method
def isoneutral_friction(vs):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    if vs.enable_implicit_vert_friction:
        aloc = vs.u[:, :, :, vs.taup1]
    else:
        aloc = vs.u[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[2:-1, 1:-2, :])
    delta, a_tri, b_tri, c_tri = (
        allocate(vs, ('xu', 'yt', 'zt'))[1:-2, 1:-2]
        for _ in range(4)
    )
    delta = jax.ops.index_update(delta, jax.ops.index[:, :, :-1],
        vs.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    delta = jax.ops.index_update(delta, jax.ops.index[-1],
        0.)
    a_tri = jax.ops.index_update(a_tri, jax.ops.index[:, :, 1:],
        -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, 1:-1],
        1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, -1],
        1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = jax.ops.index_update(c_tri, jax.ops.index[...],
        - delta / vs.dzt[np.newaxis, np.newaxis, :])
    sol, water_mask = utilities.solve_implicit(
        vs, ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge
    )
    vs.u = jax.ops.index_update(vs.u, jax.ops.index[1:-2, 1:-2, :, vs.taup1],
        utilities.where(vs, water_mask, sol, vs.u[1:-2, 1:-2, :, vs.taup1]))
    vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[1:-2, 1:-2, :],
        (vs.u[1:-2, 1:-2, :, vs.taup1] \
                                - aloc[1:-2, 1:-2, :]) / vs.dt_mom * water_mask)

    if vs.enable_conserve_energy:
        # diagnose dissipation
        diss = allocate(vs, ('xu', 'yt', 'zt'))
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[2:-1, 1:-2, :-1])
        vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
            fxa * (vs.u[1:-2, 1:-2, 1:, vs.taup1] - vs.u[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1]
        )
        diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
            (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
            * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1]
        )
        diss = jax.ops.index_update(diss, jax.ops.index[:, :, -1], 0.)
        diss = numerics.ugrid_to_tgrid(vs, diss)
        vs.K_diss_gm = jax.ops.index_update(vs.K_diss_gm, jax.ops.index[...], diss)

    if vs.enable_implicit_vert_friction:
        aloc = vs.v[:, :, :, vs.taup1]
    else:
        aloc = vs.v[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (allocate(vs, ('xt', 'yu', 'zt'))[1:-2, 1:-2] for _ in range(4))
    delta = jax.ops.index_update(delta, jax.ops.index[:, :, :-1],
        vs.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    delta = jax.ops.index_update(delta, jax.ops.index[-1],
        0.)
    a_tri = jax.ops.index_update(a_tri, jax.ops.index[:, :, 1:],
        -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, 1:-1],
        1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, -1],
        1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = jax.ops.index_update(c_tri, jax.ops.index[...],
        - delta / vs.dzt[np.newaxis, np.newaxis, :])
    sol, water_mask = utilities.solve_implicit(
        vs, ks, a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], b_edge=b_tri_edge
    )
    vs.v = jax.ops.index_update(vs.v, jax.ops.index[1:-2, 1:-2, :, vs.taup1],
        utilities.where(vs, water_mask, sol, vs.v[1:-2, 1:-2, :, vs.taup1]))
    vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[1:-2, 1:-2, :],
        (vs.v[1:-2, 1:-2, :, vs.taup1] -
                                 aloc[1:-2, 1:-2, :]) / vs.dt_mom * water_mask
    )

    if vs.enable_conserve_energy:
        # diagnose dissipation
        diss = allocate(vs, ('xt', 'yu', 'zt'))
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[1:-2, 2:-1, :-1])
        vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
            fxa * (vs.v[1:-2, 1:-2, 1:, vs.taup1] - vs.v[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1]
        )
        diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
            (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
            * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
        diss = jax.ops.index_update(diss, jax.ops.index[:, :, -1], 0.)
        diss = numerics.vgrid_to_tgrid(vs, diss)
        vs.K_diss_gm += diss
