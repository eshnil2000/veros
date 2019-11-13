import math

from .. import veros_method, runtime_settings as rs
from ..variables import allocate
from . import numerics, utilities


@veros_method
def explicit_vert_friction(vs):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """
    diss = allocate(vs, ('xt', 'yt', 'zw'))

    """
    vertical friction of zonal momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
        fxa * (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[:, :, -1],
        0.0)
    vs.du_mix = jax.ops.index_update(vs.du_mix, jax.ops.index[:, :, 0],
        vs.flux_top[:, :, 0] / vs.dzt[0] * vs.maskU[:, :, 0])
    vs.du_mix = jax.ops.index_update(vs.du_mix, jax.ops.index[:, :, 1:],
        (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) / vs.dzt[1:] * vs.maskU[:, :, 1:])

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
        (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
    diss = jax.ops.index_update(diss, jax.ops.index[:, :, vs.nz - 1],
        0.0)
    diss = jax.ops.index_update(diss, jax.ops.index[...],
        numerics.ugrid_to_tgrid(vs, diss))
    vs.K_diss_v += diss

    """
    vertical friction of meridional momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
        fxa * (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
        / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskV[1:-2, 1:-2, 1:] \
        * vs.maskV[1:-2, 1:-2, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[:, :, -1],
        0.0)
    vs.dv_mix = jax.ops.index_update(vs.dv_mix, jax.ops.index[:, :, 1:],
        (vs.flux_top[:, :, 1:] - vs.flux_top[:, :, :-1]) \
        / vs.dzt[np.newaxis, np.newaxis, 1:] * vs.maskV[:, :, 1:])
    vs.dv_mix = jax.ops.index_update(vs.dv_mix, jax.ops.index[:, :, 0],
        vs.flux_top[:, :, 0] / vs.dzt[0] * vs.maskV[:, :, 0])

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
        (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
        * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
    diss = jax.ops.index_update(diss, jax.ops.index[:, :, -1],
        0.0)
    diss = jax.ops.index_update(diss, jax.ops.index[...],
        numerics.vgrid_to_tgrid(vs, diss))
    vs.K_diss_v += diss


@veros_method
def implicit_vert_friction(vs):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """
    a_tri = allocate(vs, ('xt', 'yt', 'zw'))[1:-2, 1:-2]
    b_tri = allocate(vs, ('xt', 'yt', 'zw'))[1:-2, 1:-2]
    c_tri = allocate(vs, ('xt', 'yt', 'zw'))[1:-2, 1:-2]
    d_tri = allocate(vs, ('xt', 'yt', 'zw'))[1:-2, 1:-2]
    delta = allocate(vs, ('xt', 'yt', 'zw'))[1:-2, 1:-2]
    diss = allocate(vs, ('xt', 'yt', 'zw'))

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[2:-1, 1:-2]) - 1
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    delta = jax.ops.index_update(delta, jax.ops.index[:, :, :-1],
        vs.dt_mom / vs.dzw[:-1] * fxa * \
        vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    a_tri = jax.ops.index_update(a_tri, jax.ops.index[:, :, 1:],
        -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, 1:],
        1 + delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = jax.ops.index_add(b_tri, jax.ops.index[:, :, 1:-1],
        delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri = jax.ops.index_update(c_tri, jax.ops.index[...],
        -delta / vs.dzt[np.newaxis, np.newaxis, :])
    d_tri = jax.ops.index_update(d_tri, jax.ops.index[...],
        vs.u[1:-2, 1:-2, :, vs.tau])
    res, mask = utilities.solve_implicit(vs, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    vs.u = jax.ops.index_update(vs.u, jax.ops.index[1:-2, 1:-2, :, vs.taup1],
        utilities.where(vs, mask, res, vs.u[1:-2, 1:-2, :, vs.taup1]))

    vs.du_mix = jax.ops.index_update(vs.du_mix, jax.ops.index[1:-2, 1:-2],
        (vs.u[1:-2, 1:-2, :, vs.taup1] -
                                vs.u[1:-2, 1:-2, :, vs.tau]) / vs.dt_mom)

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
        fxa * (vs.u[1:-2, 1:-2, 1:, vs.taup1] - vs.u[1:-2, 1:-2, :-1, vs.taup1]) \
        / vs.dzw[:-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
        (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[:-1])
    diss = jax.ops.index_update(diss, jax.ops.index[:, :, -1],
        0.0)
    diss = jax.ops.index_update(diss, jax.ops.index[...],
        numerics.ugrid_to_tgrid(vs, diss))
    vs.K_diss_v += diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[1:-2, 2:-1]) - 1
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    delta = jax.ops.index_update(delta, jax.ops.index[:, :, :-1],
        vs.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    a_tri = jax.ops.index_update(a_tri, jax.ops.index[:, :, 1:],
        -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = jax.ops.index_update(b_tri, jax.ops.index[:, :, 1:],
        1 + delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = jax.ops.index_add(b_tri, jax.ops.index[:, :, 1:-1], delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri = jax.ops.index_update(c_tri, jax.ops.index[:, :, :-1],
        -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1])
    c_tri = jax.ops.index_update(c_tri, jax.ops.index[:, :, -1],
        0.)
    d_tri = jax.ops.index_update(d_tri, jax.ops.index[...],
        vs.v[1:-2, 1:-2, :, vs.tau])
    res, mask = utilities.solve_implicit(vs, kss, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    vs.v = jax.ops.index_update(vs.v, jax.ops.index[1:-2, 1:-2, :, vs.taup1],
        utilities.where(vs, mask, res, vs.v[1:-2, 1:-2, :, vs.taup1]))
    vs.dv_mix = jax.ops.index_update(vs.dv_mix, jax.ops.index[1:-2, 1:-2],
        (vs.v[1:-2, 1:-2, :, vs.taup1] - vs.v[1:-2, 1:-2, :, vs.tau]) / vs.dt_mom)

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    vs.flux_top = jax.ops.index_update(vs.flux_top, jax.ops.index[1:-2, 1:-2, :-1],
        fxa * (vs.v[1:-2, 1:-2, 1:, vs.taup1] - vs.v[1:-2, 1:-2, :-1, vs.taup1]) \
        / vs.dzw[:-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 1:-2, :-1],
        (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
                             * vs.flux_top[1:-2, 1:-2, :-1] / vs.dzw[:-1])
    diss = jax.ops.index_update(diss, jax.ops.index[:, :, -1],
        0.0)
    diss = numerics.vgrid_to_tgrid(vs, diss)
    vs.K_diss_v += diss


@veros_method
def rayleigh_friction(vs):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    vs.du_mix += -vs.maskU * vs.r_ray * vs.u[..., vs.tau]
    if vs.enable_conserve_energy:
        diss = vs.maskU * vs.r_ray * vs.u[..., vs.tau]**2
        vs.K_diss_bot += numerics.calc_diss(vs, diss, 'U')
    vs.dv_mix += -vs.maskV * vs.r_ray * vs.v[..., vs.tau]
    if vs.enable_conserve_energy:
        diss = vs.maskV * vs.r_ray * vs.v[..., vs.tau]**2
        vs.K_diss_bot += numerics.calc_diss(vs, diss, 'V')


@veros_method
def linear_bottom_friction(vs):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    if vs.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = np.arange(vs.nz) == k[:, :, np.newaxis]
        vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[1:-2, 2:-2],
            -(vs.maskU[1:-2, 2:-2] * vs.r_bot_var_u[1:-2, 2:-2, np.newaxis])
            * vs.u[1:-2, 2:-2, :, vs.tau] * mask)
        if vs.enable_conserve_energy:
            diss = allocate(vs, ('xt', 'yt', 'zt'))
            diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 2:-2],
                vs.maskU[1:-2, 2:-2] * vs.r_bot_var_u[1:-2, 2:-2, np.newaxis] \
                                    * vs.u[1:-2, 2:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot += numerics.calc_diss(vs, diss, 'U')

        k = np.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(vs.nz) == k[:, :, np.newaxis]
        vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[2:-2, 1:-2],
            -(vs.maskV[2:-2, 1:-2] * vs.r_bot_var_v[2:-2, 1:-2, np.newaxis])
            * vs.v[2:-2, 1:-2, :, vs.tau] * mask)
        if vs.enable_conserve_energy:
            diss = allocate(vs, ('xt', 'yu', 'zt'))
            diss = jax.ops.index_update(diss, jax.ops.index[2:-2, 1:-2],
                vs.maskV[2:-2, 1:-2] * vs.r_bot_var_v[2:-2, 1:-2, np.newaxis] \
                                    * vs.v[2:-2, 1:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot += numerics.calc_diss(vs, diss, 'V')
    else:
        """
        with constant coefficient
        """
        k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = np.arange(vs.nz) == k[:, :, np.newaxis]
        vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[1:-2, 2:-2],
            -vs.maskU[1:-2, 2:-2] * vs.r_bot * vs.u[1:-2, 2:-2, :, vs.tau] * mask)
        if vs.enable_conserve_energy:
            diss = allocate(vs, ('xt', 'yt', 'zt'))
            diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 2:-2],
                vs.maskU[1:-2, 2:-2] * vs.r_bot * vs.u[1:-2, 2:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot += numerics.calc_diss(vs, diss, 'U')

        k = np.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(vs.nz) == k[:, :, np.newaxis]
        vs.dv_mix = jax.ops.index_update(vs.dv_mix, jax.ops.index[2:-2, 1:-2],
            -vs.maskV[2:-2, 1:-2] * vs.r_bot * vs.v[2:-2, 1:-2, :, vs.tau] * mask)
        if vs.enable_conserve_energy:
            diss = allocate(vs, ('xt', 'yu', 'zt'))
            diss = jax.ops.index_update(diss, jax.ops.index[2:-2, 1:-2],
                vs.maskV[2:-2, 1:-2] * vs.r_bot * vs.v[2:-2, 1:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot += numerics.calc_diss(vs, diss, 'V')


@veros_method
def quadratic_bottom_friction(vs):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    # we might want to account for EKE in the drag, also a tidal residual
    k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
    mask = k[..., np.newaxis] == np.arange(vs.nz)[np.newaxis, np.newaxis, :]
    fxa = vs.maskV[1:-2, 2:-2, :] * vs.v[1:-2, 2:-2, :, vs.tau]**2 \
        + vs.maskV[1:-2, 1:-3, :] * vs.v[1:-2, 1:-3, :, vs.tau]**2 \
        + vs.maskV[2:-1, 2:-2, :] * vs.v[2:-1, 2:-2, :, vs.tau]**2 \
        + vs.maskV[2:-1, 1:-3, :] * vs.v[2:-1, 1:-3, :, vs.tau]**2
    fxa = np.sqrt(vs.u[1:-2, 2:-2, :, vs.tau]**2 + 0.25 * fxa)
    aloc = vs.maskU[1:-2, 2:-2, :] * vs.r_quad_bot * vs.u[1:-2, 2:-2, :, vs.tau] \
        * fxa / vs.dzt[np.newaxis, np.newaxis, :] * mask
    vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[1:-2, 2:-2, :], -aloc)

    if vs.enable_conserve_energy:
        diss = allocate(vs, ('xt', 'yt', 'zt'))
        diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 2:-2, :],
            aloc * vs.u[1:-2, 2:-2, :, vs.tau])
        vs.K_diss_bot = jax.ops.index_add(vs.K_diss_bot, jax.ops.index[...],
            numerics.calc_diss(vs, diss, 'U'))

    k = np.maximum(vs.kbot[2:-2, 1:-2], vs.kbot[2:-2, 2:-1]) - 1
    mask = k[..., np.newaxis] == np.arange(vs.nz)[np.newaxis, np.newaxis, :]
    fxa = vs.maskU[2:-2, 1:-2, :] * vs.u[2:-2, 1:-2, :, vs.tau]**2 \
        + vs.maskU[1:-3, 1:-2, :] * vs.u[1:-3, 1:-2, :, vs.tau]**2 \
        + vs.maskU[2:-2, 2:-1, :] * vs.u[2:-2, 2:-1, :, vs.tau]**2 \
        + vs.maskU[1:-3, 2:-1, :] * vs.u[1:-3, 2:-1, :, vs.tau]**2
    fxa = np.sqrt(vs.v[2:-2, 1:-2, :, vs.tau]**2 + 0.25 * fxa)
    aloc = vs.maskV[2:-2, 1:-2, :] * vs.r_quad_bot * vs.v[2:-2, 1:-2, :, vs.tau] \
        * fxa / vs.dzt[np.newaxis, np.newaxis, :] * mask
    vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[2:-2, 1:-2, :],
        -aloc)

    if vs.enable_conserve_energy:
        diss = allocate(vs, ('xt', 'yu', 'zt'))
        diss = jax.ops.index_update(diss, jax.ops.index[2:-2, 1:-2, :],
            aloc * vs.v[2:-2, 1:-2, :, vs.tau])
        vs.K_diss_bot = jax.ops.index_add(vs.K_diss_bot, jax.ops.index[...],
            numerics.calc_diss(vs, diss, 'V'))


@veros_method
def harmonic_friction(vs):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    diss = allocate(vs, ('xt', 'yt', 'zt'))

    """
    Zonal velocity
    """
    if vs.enable_hor_friction_cos_scaling:
        fxa = vs.cost**vs.hor_friction_cosPower
        vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1],
            vs.A_h * fxa[np.newaxis, :, np.newaxis] * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
                / (vs.cost * vs.dxt[1:, np.newaxis])[:, :, np.newaxis] * vs.maskU[1:] * vs.maskU[:-1])
        fxa = vs.cosu**vs.hor_friction_cosPower
        vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1],
            vs.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
                / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
        if vs.enable_noslip_lateral:
             vs.flux_north = jax.ops.index_add(vs.flux_north, jax.ops.index[:, :-1],
                2 * vs.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, 1:, :, vs.tau]) \
                / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * vs.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, :-1, :, vs.tau]) \
                / vs.dyu[np.newaxis, :-1, np.newaxis] * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
    else:
        vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1, :, :],
            vs.A_h * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
            / (vs.cost * vs.dxt[1:, np.newaxis])[:, :, np.newaxis] * vs.maskU[1:] * vs.maskU[:-1])
        vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1, :],
            vs.A_h * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
        if vs.enable_noslip_lateral:
             vs.flux_north = jax.ops.index_add(vs.flux_north, jax.ops.index[:, :-1],
                2 * vs.A_h * vs.u[:, 1:, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
                * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * vs.A_h * vs.u[:, :-1, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
                * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])

    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    """
    update tendency
    """
    vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[2:-2, 2:-2, :],
        vs.maskU[2:-2, 2:-2] * ((vs.flux_east[2:-2, 2:-2] - vs.flux_east[1:-3, 2:-2])
                                                              / (vs.cost[2:-2] * vs.dxu[2:-2, np.newaxis])[:, :, np.newaxis]
                                                              + (vs.flux_north[2:-2, 2:-2] - vs.flux_north[2:-2, 1:-3])
                                                              / (vs.cost[2:-2] * vs.dyt[2:-2])[np.newaxis, :, np.newaxis])
    )

    if vs.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 2:-2],
            0.5 * ((vs.u[2:-1, 2:-2, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * vs.flux_east[1:-2, 2:-2]
                                + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[:-3, 2:-2, :, vs.tau]) * vs.flux_east[:-3, 2:-2]) \
            / (vs.cost[2:-2] * vs.dxu[1:-2, np.newaxis])[:, :, np.newaxis]\
            + 0.5 * ((vs.u[1:-2, 3:-1, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * vs.flux_north[1:-2, 2:-2]
                   + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[1:-2, 1:-3, :, vs.tau]) * vs.flux_north[1:-2, 1:-3]) \
            / (vs.cost[2:-2] * vs.dyt[2:-2])[np.newaxis, :, np.newaxis]
        )
        vs.K_diss_h = jax.ops.index_update(vs.K_diss_h, jax.ops.index[...], 0.)
        vs.K_diss_h = jax.ops.index_add(vs.K_diss_h, jax.ops.index[...],
            numerics.calc_diss(vs, diss, 'U'))

    """
    Meridional velocity
    """
    if vs.enable_hor_friction_cos_scaling:
        vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1],
            vs.A_h * vs.cosu[np.newaxis, :, np.newaxis] ** vs.hor_friction_cosPower \
            * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
            / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * vs.maskV[:-1])
        if vs.enable_noslip_lateral:
            vs.flux_east = jax.ops.index_add(vs.flux_east, jax.ops.index[:-1],
                2 * vs.A_h * fxa[np.newaxis, :, np.newaxis] * vs.v[1:, :, :, vs.tau] \
                / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * (1 - vs.maskV[:-1]) \
                - 2 * vs.A_h * fxa[np.newaxis, :, np.newaxis] * vs.v[:-1, :, :, vs.tau] \
                / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * (1 - vs.maskV[1:]) * vs.maskV[:-1])

        vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1],
            vs.A_h * vs.cost[np.newaxis, 1:, np.newaxis] ** vs.hor_friction_cosPower \
            * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
            / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] * vs.maskV[:, :-1] * vs.maskV[:, 1:])
    else:
        vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1],
            vs.A_h * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
            / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * vs.maskV[:-1])
        if vs.enable_noslip_lateral:
            vs.flux_east = jax.ops.index_add(vs.flux_east, jax.ops.index[:-1],
                2 * vs.A_h * vs.v[1:, :, :, vs.tau] / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * vs.maskV[1:] * (1 - vs.maskV[:-1]) \
                - 2 * vs.A_h * vs.v[:-1, :, :, vs.tau] / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * (1 - vs.maskV[1:]) * vs.maskV[:-1])
        vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1],
            vs.A_h * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
            / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] * vs.maskV[:, :-1] * vs.maskV[:, 1:])
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    """
    update tendency
    """
    vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[2:-2, 2:-2],
        vs.maskV[2:-2, 2:-2] * ((vs.flux_east[2:-2, 2:-2] - vs.flux_east[1:-3, 2:-2])
                                                   / (vs.cosu[2:-2] * vs.dxt[2:-2, np.newaxis])[:, :, np.newaxis]
                                                   + (vs.flux_north[2:-2, 2:-2] - vs.flux_north[2:-2, 1:-3])
                                                   / (vs.dyu[2:-2] * vs.cosu[2:-2])[np.newaxis, :, np.newaxis])
    )

    if vs.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss = jax.ops.index_update(diss, jax.ops.index[2:-2, 1:-2],
            0.5 * ((vs.v[3:-1, 1:-2, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * vs.flux_east[2:-2, 1:-2]
                                + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[1:-3, 1:-2, :, vs.tau]) * vs.flux_east[1:-3, 1:-2]) \
            / (vs.cosu[1:-2] * vs.dxt[2:-2, np.newaxis])[:, :, np.newaxis] \
            + 0.5 * ((vs.v[2:-2, 2:-1, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * vs.flux_north[2:-2, 1:-2]
                   + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[2:-2, :-3, :, vs.tau]) * vs.flux_north[2:-2, :-3]) \
            / (vs.cosu[1:-2] * vs.dyu[1:-2])[np.newaxis, :, np.newaxis]
        )
        vs.K_diss_h = jax.ops.index_add(vs.K_diss_h, jax.ops.index[...], numerics.calc_diss(vs, diss, 'V'))


@veros_method
def biharmonic_friction(vs):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    fxa = math.sqrt(abs(vs.A_hbi))

    """
    Zonal velocity
    """
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1, :, :],
        fxa * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.maskU[1:, :, :] * vs.maskU[:-1, :, :]
    )
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1, :],
        fxa * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:, :] \
        * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis]
    )
    if vs.enable_noslip_lateral:
        vs.flux_north = jax.ops.index_add(vs.flux_north, jax.ops.index[:, :-1],
            2 * fxa * vs.u[:, 1:, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
            - 2 * fxa * vs.u[:, :-1, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis]
        )
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    del2 = allocate(vs, ('xt', 'yt', 'zt'))
    del2 = jax.ops.index_update(del2, jax.ops.index[1:, 1:, :],
        (vs.flux_east[1:, 1:, :] - vs.flux_east[:-1, 1:, :]) \
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxu[1:, np.newaxis, np.newaxis]) \
        + (vs.flux_north[1:, 1:, :] - vs.flux_north[1:, :-1, :]) \
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis])
    )

    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1, :, :],
        fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.maskU[1:, :, :] * vs.maskU[:-1, :, :]
    )
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1, :],
        fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:, :] \
        * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis]
    )
    if vs.enable_noslip_lateral:
        vs.flux_north = jax.ops.index_add(vs.flux_north, jax.ops.index[:,:-1,:],
            2 * fxa * del2[:, 1:, :] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * vs.maskU[:, 1:, :] * (1 - vs.maskU[:, :-1, :]) * vs.cosu[np.newaxis, :-1, np.newaxis] \
            - 2 * fxa * del2[:, :-1, :] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - vs.maskU[:, 1:, :]) * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis]
        )
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    """
    update tendency
    """
    vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[2:-2, 2:-2, :],
        -vs.maskU[2:-2, 2:-2, :] * ((vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                          / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[2:-2, np.newaxis, np.newaxis])
                                                          + (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                          / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
    )
    if vs.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        utilities.enforce_boundaries(vs, vs.flux_east)
        utilities.enforce_boundaries(vs, vs.flux_north)
        diss = allocate(vs, ('xt', 'yt', 'zt'))
        diss = jax.ops.index_update(diss, jax.ops.index[1:-2, 2:-2, :],
            -0.5 * ((vs.u[2:-1, 2:-2, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * vs.flux_east[1:-2, 2:-2, :]
                                    + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[:-3, 2:-2, :, vs.tau]) * vs.flux_east[:-3, 2:-2, :]) \
            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[1:-2, np.newaxis, np.newaxis])  \
            - 0.5 * ((vs.u[1:-2, 3:-1, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * vs.flux_north[1:-2, 2:-2, :]
                   + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[1:-2, 1:-3, :, vs.tau]) * vs.flux_north[1:-2, 1:-3, :]) \
            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])
        )
        vs.K_diss_h = jax.ops.index_update(vs.K_diss_h, jax.ops.index[...], 0.)
        vs.K_diss_h = jax.ops.index_add(vs.K_diss_h, jax.ops.index[...], numerics.calc_diss(vs, diss, 'U'))

    """
    Meridional velocity
    """
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1, :, :],
        fxa * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
        / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
        * vs.maskV[1:, :, :] * vs.maskV[:-1, :, :]
    )
    if vs.enable_noslip_lateral:
        vs.flux_east = jax.ops.index_add(vs.flux_east, jax.ops.index[:-1, :, :],
        2 * fxa * vs.v[1:, :, :, vs.tau] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskV[1:, :, :] * (1 - vs.maskV[:-1, :, :]) \
            - 2 * fxa * vs.v[:-1, :, :, vs.tau] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - vs.maskV[1:, :, :]) * vs.maskV[:-1, :, :]
        )
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1, :],
        fxa * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] \
        * vs.maskV[:, :-1, :] * vs.maskV[:, 1:, :]
    )
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    del2 = jax.ops.index_update(del2, jax.ops.index[1:, 1:, :],
        (vs.flux_east[1:, 1:, :] - vs.flux_east[:-1, 1:, :]) \
        / (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis])  \
        + (vs.flux_north[1:, 1:, :] - vs.flux_north[1:, :-1, :]) \
        / (vs.dyu[np.newaxis, 1:, np.newaxis] * vs.cosu[np.newaxis, 1:, np.newaxis])
    )

    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[:-1, :, :],
        fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
        * vs.maskV[1:, :, :] * vs.maskV[:-1, :, :]
    )
    if vs.enable_noslip_lateral:
        vs.flux_east = jax.ops.index_add(vs.flux_east, jax.ops.index[:-1, :, :],
            2 * fxa * del2[1:, :, :] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskV[1:, :, :] * (1 - vs.maskV[:-1, :, :]) \
            - 2 * fxa * del2[:-1, :, :] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - vs.maskV[1:, :, :]) * vs.maskV[:-1, :, :]
        )
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, :-1, :],
        fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] \
        * vs.maskV[:, :-1, :] * vs.maskV[:, 1:, :]
    )
    vs.flux_east = jax.ops.index_update(vs.flux_east, jax.ops.index[-1, :, :], 0.)
    vs.flux_north = jax.ops.index_update(vs.flux_north, jax.ops.index[:, -1, :], 0.)

    """
    update tendency
    """
    vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[2:-2, 2:-2, :],
        -vs.maskV[2:-2, 2:-2, :] * ((vs.flux_east[2:-2, 2:-2, :] - vs.flux_east[1:-3, 2:-2, :])
                                                            / (vs.cosu[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                            + (vs.flux_north[2:-2, 2:-2, :] - vs.flux_north[2:-2, 1:-3, :])
                                                            / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis]))
    )

    if vs.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        utilities.enforce_boundaries(vs, vs.flux_east)
        utilities.enforce_boundaries(vs, vs.flux_north)
        diss = jax.ops.index_update(diss, jax.ops.index[2:-2, 1:-2, :],
            -0.5 * ((vs.v[3:-1, 1:-2, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * vs.flux_east[2:-2, 1:-2, :]
                                    + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[1:-3, 1:-2, :, vs.tau]) * vs.flux_east[1:-3, 1:-2, :]) \
            / (vs.cosu[np.newaxis, 1:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis]) \
            - 0.5 * ((vs.v[2:-2, 2:-1, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * vs.flux_north[2:-2, 1:-2, :]
                   + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[2:-2, :-3, :, vs.tau]) * vs.flux_north[2:-2, :-3, :]) \
            / (vs.cosu[np.newaxis, 1:-2, np.newaxis] * vs.dyu[np.newaxis, 1:-2, np.newaxis])
        )
        vs.K_diss_h = jax.ops.index_add(vs.K_diss_h, jax.ops.index[...], numerics.calc_diss(vs, diss, 'V'))


@veros_method
def momentum_sources(vs):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    vs.du_mix = jax.ops.index_add(vs.du_mix, jax.ops.index[...],
        vs.maskU * vs.u_source)
    if vs.enable_conserve_energy:
        diss = -vs.maskU * vs.u[..., vs.tau] * vs.u_source
        vs.K_diss_bot = jax.ops.index_add(vs.K_diss_bot, jax.ops.index[...], numerics.calc_diss(vs, diss, 'U'))
    vs.dv_mix = jax.ops.index_add(vs.dv_mix, jax.ops.index[...], vs.maskV * vs.v_source)
    if vs.enable_conserve_energy:
        diss = -vs.maskV * vs.v[..., vs.tau] * vs.v_source
        vs.K_diss_bot = jax.ops.index_add(vs.K_diss_bot, jax.ops.index[...], numerics.calc_diss(vs, diss, 'V'))
