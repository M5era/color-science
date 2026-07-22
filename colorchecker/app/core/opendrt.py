"""1:1 Python port of OpenDRT v1.1.0b50 (Jed Smith).

Source: reference/OpenDRT_installed.dctl — the EXACT file Marc runs in
Resolve (confirmed 2026-07-21). Transcribed formula-for-formula,
vectorized float64, following the reuleaux-port discipline: every
function mirrors its DCTL counterpart including edge behavior (sdivf
zero-guard, spowf sign passthrough, C-fmod hue wrap).

LICENSE: OpenDRT is GPLv3 (github.com/jedypod/open-display-transform).
This port is a derivative work and carries the same license.

SCOPE: the pixel pipeline is complete for Marc's confirmed
configuration, resolved from the DCTL's preset tables:

    Input Gamut      Arri Wide Gamut 3
    Input Transfer   Arri LogC3
    Look Preset      Standard          (look_preset==0 table)
    Tonescale Preset Low Contrast      (tonescale_preset==1 override)
    Creative White   USE LOOK PRESET   (-> D65: the identity CAT path)
    Display Encoding sRGB Display      (Rec.709 gamut, 2.2 power,
                                        surround=2, clamp on, Lp=100)

All look/tonescale scalars live in OpenDRTConfig, so any resolved
preset can be expressed; paths that need matrices not transcribed yet
(other input gamuts, non-D65 creative whites, HDR eotfs) raise
NotImplementedError rather than silently doing the wrong thing.

Validation gate: tests/test_opendrt.py compares this port against the
Resolve-baked 65^3 cube in test_luts/ (float32 lattice, so agreement
is expected at ~1e-3, not 1e-9).
"""

from dataclasses import dataclass, field

import numpy as np

SQRT3 = 1.73205080756887729353
PI = 3.14159265358979323846

# ---------------------------------------------------------- matrices
# (rows exactly as the DCTL's make_float3x3 rows)

MATRIX_ARRIWG3_TO_XYZ = np.array([
    [0.638007619284, 0.214703856337, 0.097744451431],
    [0.291953779, 0.823841041511, -0.11579482051],
    [0.002798279032, -0.067034235689, 1.15329370742],
])
MATRIX_P3D65_TO_XYZ = np.array([
    [0.486570948648216151, 0.265667693169093, 0.198217285234362467],
    [0.228974564069748754, 0.691738521836506193, 0.079286914093744984],
    [-4.00000000000000029e-17, 0.0451133818589026167, 1.04394436890097575],
])
MATRIX_XYZ_TO_P3D65 = np.array([
    [2.49349691194142542, -0.93138361791912383, -0.402710784450716841],
    [-0.829488969561574696, 1.76266406031834655, 0.0236246858419435941],
    [0.0358458302437844531, -0.0761723892680418041, 0.956884524007687309],
])
MATRIX_XYZ_TO_REC709 = np.array([
    [3.24096994190452348, -1.53738317757009435, -0.498610760293003552],
    [-0.969243636280879506, 1.87596750150771996, 0.0415550574071755843],
    [0.0556300796969936354, -0.20397695888897649, 1.05697151424287816],
])

_INPUT_GAMUTS = {
    "awg3": MATRIX_ARRIWG3_TO_XYZ,
    "p3d65": MATRIX_P3D65_TO_XYZ,
    "xyz": np.eye(3),
}


# ------------------------------------------------------ math helpers

def _sdiv(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(b == 0.0, 0.0, out)


def _spow(a, b):
    with np.errstate(invalid="ignore"):
        p = np.power(np.where(a > 0.0, a, 1.0), b)
    return np.where(a <= 0.0, a, p)


def compress_toe_quadratic(x, toe, inv):
    if toe == 0.0:
        return x
    x = np.asarray(x, dtype=np.float64)
    if not inv:
        return _sdiv(_spow(x, 2.0), x + toe)
    return (x + np.sqrt(np.maximum(x * (4.0 * toe + x), 0.0))) / 2.0


def compress_hyperbolic_power(x, s, p):
    return _spow(_sdiv(x, x + s), p)


def compress_toe_cubic(x, m, w, inv):
    if m == 1.0:
        return x
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    if not inv:
        return x * (x2 + m * w) / (x2 + w)
    p0 = x2 - 3.0 * m * w
    p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w
    p2 = np.power(
        np.sqrt(np.maximum(x2 * p1 * p1 - 4.0 * p0 ** 3, 0.0)) / 2.0
        + x * p1 / 2.0, 1.0 / 3.0)
    return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0


def contrast_high(x, p, pv, pv_lx):
    """Forward contrast_high (inv==0)."""
    x = np.asarray(x, dtype=np.float64)
    x0 = 0.18 * 2.0 ** pv
    if p == 1.0:
        return x
    o = x0 - x0 / p
    s0 = x0 ** (1.0 - p) / p
    x1 = x0 * 2.0 ** pv_lx
    k1 = p * s0 * x1 ** p / x1
    y1 = s0 * x1 ** p + o
    curved = s0 * _spow(x, p) + o
    out = np.where(x > x1, k1 * (x - x1) + y1, curved)
    return np.where(x < x0, x, out)


def softplus(x, s):
    x = np.asarray(x, dtype=np.float64)
    if s < 1e-4:
        return x
    inner = s * np.log(np.maximum(0.0, 1.0 + np.exp(np.minimum(x / s, 10.0))))
    return np.where(x > 10.0 * s, x, inner)


def gauss_window(x, w):
    return np.exp(-x * x / w)


def opponent(rgb):
    return (rgb[..., 0] - rgb[..., 2],
            rgb[..., 1] - (rgb[..., 0] + rgb[..., 2]) / 2.0)


def hue_offset(h, o):
    # C fmod semantics (sign of the dividend) — np.fmod matches
    return np.fmod(h - o + PI, 2.0 * PI) - PI


def oetf_arri_logc3(x):
    x = np.asarray(x, dtype=np.float64)
    lin = (x - 0.092809) / 5.367655
    # clamp the exponent so wildly out-of-range code values (a contorted
    # upstream node can push code >> 1) don't overflow 10**e — valid LogC3
    # (x in [0,1]) keeps e in ~[-1.6, 2.5], far inside the clamp, so the
    # 1:1 port is unchanged where it matters (validation gate still passes)
    e = np.clip((x - 0.385537) / 0.247190, -50.0, 50.0)
    exp = (10.0 ** e - 0.052272) / 5.555556
    return np.where(x < 5.367655 * 0.010591 + 0.092809, lin, exp)


_OETFS = {"linear": lambda x: np.asarray(x, dtype=np.float64),
          "arri_logc3": oetf_arri_logc3}


# ------------------------------------------------------------ config

@dataclass
class OpenDRTConfig:
    """All resolved scalars of the DCTL transform. Defaults = Marc's
    confirmed setup (Standard look + Low Contrast tonescale + sRGB
    display + D65 creative white + SDR sliders at defaults)."""

    in_gamut: str = "awg3"
    in_oetf: str = "arri_logc3"

    # float sliders
    tn_Lp: float = 100.0
    tn_gb: float = 0.13
    pt_hdr: float = 0.5
    tn_Lg: float = 10.0

    # tonescale (Low Contrast override of the Standard look)
    tn_con: float = 1.4
    tn_sh: float = 0.5
    tn_toe: float = 0.003
    tn_off: float = 0.005
    tn_hcon_enable: bool = False
    tn_hcon: float = 0.0
    tn_hcon_pv: float = 1.0
    tn_hcon_st: float = 4.0
    tn_lcon_enable: bool = False
    tn_lcon: float = 0.0
    tn_lcon_w: float = 0.5

    # creative white: 2 == D65 (the only transcribed CAT path)
    cwp: int = 2
    cwp_lm: float = 0.25

    # rendering space
    rs_sa: float = 0.35
    rs_rw: float = 0.25
    rs_bw: float = 0.55

    # purity compression (Standard look)
    pt_lml: float = 0.25
    pt_lml_r: float = 0.5
    pt_lml_g: float = 0.0
    pt_lml_b: float = 0.1
    pt_lmh: float = 0.25
    pt_lmh_r: float = 0.5
    pt_lmh_b: float = 0.0
    ptl_enable: bool = True
    ptl_c: float = 0.06
    ptl_m: float = 0.08
    ptl_y: float = 0.06
    ptm_enable: bool = True
    ptm_low: float = 0.4
    ptm_low_rng: float = 0.25
    ptm_low_st: float = 0.5
    ptm_high: float = -0.8
    ptm_high_rng: float = 0.35
    ptm_high_st: float = 0.4

    # brilliance
    brl_enable: bool = True
    brl: float = 0.0
    brl_r: float = -2.5
    brl_g: float = -1.5
    brl_b: float = -1.5
    brl_rng: float = 0.5
    brl_st: float = 0.35
    brlp_enable: bool = True
    brlp: float = -0.5
    brlp_r: float = -1.25
    brlp_g: float = -1.25
    brlp_b: float = -0.25

    # hue contrast / hue shift
    hc_enable: bool = True
    hc_r: float = 1.0
    hc_r_rng: float = 0.3
    hs_rgb_enable: bool = True
    hs_r: float = 0.6
    hs_r_rng: float = 0.6
    hs_g: float = 0.35
    hs_g_rng: float = 1.0
    hs_b: float = 0.66
    hs_b_rng: float = 1.0
    hs_cmy_enable: bool = True
    hs_c: float = 0.25
    hs_c_rng: float = 1.0
    hs_m: float = 0.0
    hs_m_rng: float = 1.0
    hs_y: float = 0.0
    hs_y_rng: float = 1.0

    # display encoding (sRGB Display preset)
    tn_su: int = 2
    display_gamut: int = 0    # Rec.709
    eotf: int = 1             # 2.2 power
    clamp: bool = True


MARC_CONFIG = OpenDRTConfig()


# --------------------------------------------------------- transform

def opendrt_transform(rgb: np.ndarray,
                      cfg: OpenDRTConfig = MARC_CONFIG) -> np.ndarray:
    """The DCTL transform() for (..., 3) input, vectorized float64."""
    if cfg.display_gamut != 0:
        raise NotImplementedError("only Rec.709 display gamut transcribed")
    if not (0 < cfg.eotf < 4):
        raise NotImplementedError("only power-law display EOTFs transcribed")
    if cfg.cwp != 2:
        raise NotImplementedError("only the D65 creative white transcribed")

    rgb = np.asarray(rgb, dtype=np.float64)

    # tonescale constraint pre-calculations (all scalars)
    ts_x1 = 2.0 ** (6.0 * cfg.tn_sh + 4.0)
    ts_y1 = cfg.tn_Lp / 100.0
    ts_x0 = 0.18 + cfg.tn_off
    ts_y0 = cfg.tn_Lg / 100.0 * (1.0 + cfg.tn_gb * np.log2(ts_y1))
    ts_s0 = float(compress_toe_quadratic(ts_y0, cfg.tn_toe, 1))
    ts_p = cfg.tn_con / (1.0 + cfg.tn_su * 0.05)
    ts_s10 = ts_x0 * (ts_s0 ** (-1.0 / cfg.tn_con) - 1.0)
    ts_m1 = ts_y1 / (ts_x1 / (ts_x1 + ts_s10)) ** cfg.tn_con
    ts_m2 = float(compress_toe_quadratic(ts_m1, cfg.tn_toe, 1))
    ts_s = ts_x0 * ((ts_s0 / ts_m2) ** (-1.0 / cfg.tn_con) - 1.0)
    ts_dsc = 100.0 / cfg.tn_Lp

    pt_cmp_Lf = cfg.pt_hdr * min(1.0, (cfg.tn_Lp - 100.0) / 900.0)
    s_Lp100 = ts_x0 * ((cfg.tn_Lg / 100.0) ** (-1.0 / cfg.tn_con) - 1.0)
    ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0 - pt_cmp_Lf)

    # linearize + input gamut -> P3D65
    rgb = _OETFS[cfg.in_oetf](rgb)
    in_to_xyz = _INPUT_GAMUTS[cfg.in_gamut]
    rgb = rgb @ in_to_xyz.T
    rgb = rgb @ MATRIX_XYZ_TO_P3D65.T

    # rendering space "desaturation"
    rs_w = np.array([cfg.rs_rw, 1.0 - cfg.rs_rw - cfg.rs_bw, cfg.rs_bw])
    sat_L = rgb @ rs_w
    rgb = sat_L[..., None] * cfg.rs_sa + rgb * (1.0 - cfg.rs_sa)

    # offset
    rgb = rgb + cfg.tn_off

    # tonescale norm + rgb ratios
    tsn = np.sqrt(np.maximum(0.0, (rgb * rgb).sum(axis=-1))) / SQRT3
    rgb = _sdiv(rgb, tsn[..., None])

    opp_x, opp_y = opponent(rgb)
    ach_d = np.sqrt(np.maximum(0.0, opp_x * opp_x + opp_y * opp_y)) / 2.0
    ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25, 0)

    hue = np.fmod(np.arctan2(opp_x, opp_y) + PI + 1.10714931, 2.0 * PI)

    ha_rgb = np.stack([
        gauss_window(hue_offset(hue, 0.1), 0.66),
        gauss_window(hue_offset(hue, 4.3), 0.66),
        gauss_window(hue_offset(hue, 2.3), 0.66),
    ], axis=-1)
    ha_rgb_hs = np.stack([
        gauss_window(hue_offset(hue, -0.4), 0.66),
        ha_rgb[..., 1],
        gauss_window(hue_offset(hue, 2.5), 0.66),
    ], axis=-1)
    ha_cmy = np.stack([
        gauss_window(hue_offset(hue, 3.3), 0.5),
        gauss_window(hue_offset(hue, 1.3), 0.5),
        gauss_window(hue_offset(hue, -1.15), 0.5),
    ], axis=-1)

    # brilliance (pre-tonescale)
    if cfg.brl_enable:
        brl_tsf = _spow(tsn / (tsn + 1.0), 1.0 - cfg.brl_rng)
        brl_exf = (cfg.brl + cfg.brl_r * ha_rgb[..., 0]
                   + cfg.brl_g * ha_rgb[..., 1]
                   + cfg.brl_b * ha_rgb[..., 2]) * _spow(ach_d, 1.0 / cfg.brl_st)
        brl_ex = 2.0 ** (brl_exf * np.where(brl_exf < 0.0, brl_tsf,
                                            1.0 - brl_tsf))
        tsn = tsn * brl_ex

    # contrast low
    if cfg.tn_lcon_enable:
        lcon_m = 2.0 ** (-cfg.tn_lcon)
        lcon_w = (cfg.tn_lcon_w / 4.0) ** 2
        lcon_cnst_sc = float(compress_toe_cubic(ts_x0, lcon_m, lcon_w, 1)) / ts_x0
        tsn = compress_toe_cubic(tsn * lcon_cnst_sc, lcon_m, lcon_w, 0)

    # contrast high
    if cfg.tn_hcon_enable:
        hcon_p = 2.0 ** cfg.tn_hcon
        tsn = contrast_high(tsn, hcon_p, cfg.tn_hcon_pv, cfg.tn_hcon_st)

    # hyperbolic compression
    tsn_pt = compress_hyperbolic_power(tsn, ts_s1, ts_p)
    tsn_const = compress_hyperbolic_power(tsn, s_Lp100, ts_p)
    tsn = compress_hyperbolic_power(tsn, ts_s, ts_p)

    # hue contrast R
    if cfg.hc_enable:
        hc_ts = 1.0 - tsn_const
        hc_c = hc_ts * (1.0 - ach_d) + ach_d * (1.0 - hc_ts)
        hc_c = hc_c * ach_d * ha_rgb[..., 0]
        hc_ts = _spow(hc_ts, 1.0 / cfg.hc_r_rng)
        hc_f = cfg.hc_r * (hc_c - 2.0 * hc_c * hc_ts) + 1.0
        rgb = np.stack([rgb[..., 0], rgb[..., 1] * hc_f,
                        rgb[..., 2] * hc_f], axis=-1)

    # hue shift RGB
    if cfg.hs_rgb_enable:
        hs_rgb = np.stack([
            ha_rgb_hs[..., 0] * ach_d * _spow(tsn_pt, 1.0 / cfg.hs_r_rng),
            ha_rgb_hs[..., 1] * ach_d * _spow(tsn_pt, 1.0 / cfg.hs_g_rng),
            ha_rgb_hs[..., 2] * ach_d * _spow(tsn_pt, 1.0 / cfg.hs_b_rng),
        ], axis=-1)
        hsf = np.stack([hs_rgb[..., 0] * cfg.hs_r,
                        hs_rgb[..., 1] * -cfg.hs_g,
                        hs_rgb[..., 2] * -cfg.hs_b], axis=-1)
        hsf = np.stack([hsf[..., 2] - hsf[..., 1],
                        hsf[..., 0] - hsf[..., 2],
                        hsf[..., 1] - hsf[..., 0]], axis=-1)
        rgb = rgb + hsf

    # hue shift CMY
    if cfg.hs_cmy_enable:
        compl = 1.0 - tsn_pt
        hs_cmy = np.stack([
            ha_cmy[..., 0] * ach_d * _spow(compl, 1.0 / cfg.hs_c_rng),
            ha_cmy[..., 1] * ach_d * _spow(compl, 1.0 / cfg.hs_m_rng),
            ha_cmy[..., 2] * ach_d * _spow(compl, 1.0 / cfg.hs_y_rng),
        ], axis=-1)
        hsf = np.stack([hs_cmy[..., 0] * -cfg.hs_c,
                        hs_cmy[..., 1] * cfg.hs_m,
                        hs_cmy[..., 2] * cfg.hs_y], axis=-1)
        hsf = np.stack([hsf[..., 2] - hsf[..., 1],
                        hsf[..., 0] - hsf[..., 2],
                        hsf[..., 1] - hsf[..., 0]], axis=-1)
        rgb = rgb + hsf

    # purity compression
    pt_lml_p = 1.0 + 4.0 * (1.0 - tsn_pt) * (
        cfg.pt_lml + cfg.pt_lml_r * ha_rgb_hs[..., 0]
        + cfg.pt_lml_g * ha_rgb_hs[..., 1]
        + cfg.pt_lml_b * ha_rgb_hs[..., 2])
    ptf = 1.0 - _spow(tsn_pt, pt_lml_p)
    pt_lmh_p = ((1.0 - ach_d * (cfg.pt_lmh_r * ha_rgb_hs[..., 0]
                                + cfg.pt_lmh_b * ha_rgb_hs[..., 2]))
                * (1.0 - cfg.pt_lmh * ach_d))
    ptf = _spow(ptf, pt_lmh_p)

    # mid-range purity
    if cfg.ptm_enable:
        if cfg.ptm_low_st == 0.0 or cfg.ptm_low_rng == 0.0:
            ptm_low_f = 1.0
        else:
            ptm_low_f = 1.0 + cfg.ptm_low * np.exp(
                -2.0 * ach_d * ach_d / cfg.ptm_low_st
            ) * _spow(1.0 - tsn_const, 1.0 / cfg.ptm_low_rng)
        if cfg.ptm_high_st == 0.0 or cfg.ptm_high_rng == 0.0:
            ptm_high_f = 1.0
        else:
            ptm_high_f = 1.0 + cfg.ptm_high * np.exp(
                -2.0 * ach_d * ach_d / cfg.ptm_high_st
            ) * _spow(tsn_pt, 1.0 / (4.0 * cfg.ptm_high_rng))
        ptf = ptf * ptm_low_f * ptm_high_f

    # lerp to peak achromatic in rgb ratios
    rgb = rgb * ptf[..., None] + 1.0 - ptf[..., None]

    # inverse rendering space
    sat_L = rgb @ rs_w
    rgb = (sat_L[..., None] * cfg.rs_sa - rgb) / (cfg.rs_sa - 1.0)

    # display gamut + creative whitepoint (cwp==2/D65 on Rec.709: the
    # CAT is the identity and cwp_norm==1; the P3D65->XYZ->Rec709 hop
    # remains, exactly as in display_gamut_whitepoint)
    rgb = rgb @ MATRIX_P3D65_TO_XYZ.T
    rgb = rgb @ MATRIX_XYZ_TO_REC709.T

    # post brilliance
    if cfg.brlp_enable:
        bopp_x, bopp_y = opponent(rgb)
        brlp_ach_d = np.sqrt(np.maximum(0.0, bopp_x ** 2 + bopp_y ** 2)) / 4.0
        brlp_ach_d = 1.1 * (brlp_ach_d * brlp_ach_d / (brlp_ach_d + 0.1))
        brlp_ha = ach_d[..., None] * ha_rgb
        brlp_m = (cfg.brlp + cfg.brlp_r * brlp_ha[..., 0]
                  + cfg.brlp_g * brlp_ha[..., 1]
                  + cfg.brlp_b * brlp_ha[..., 2])
        brlp_ex = 2.0 ** (brlp_m * brlp_ach_d * tsn)
        rgb = rgb * brlp_ex[..., None]

    # purity compress low
    if cfg.ptl_enable:
        rgb = np.stack([softplus(rgb[..., 0], cfg.ptl_c),
                        softplus(rgb[..., 1], cfg.ptl_m),
                        softplus(rgb[..., 2], cfg.ptl_y)], axis=-1)

    # final tonescale adjustments
    tsn = tsn * ts_m2
    tsn = compress_toe_quadratic(tsn, cfg.tn_toe, 0)
    tsn = tsn * ts_dsc

    # return from rgb ratios
    rgb = rgb * tsn[..., None]

    if cfg.clamp:
        rgb = np.clip(rgb, 0.0, 1.0)

    # inverse display EOTF (power law)
    eotf_p = 2.0 + cfg.eotf * 0.2
    rgb = _spow(rgb, 1.0 / eotf_p)
    return rgb


class OpenDRTModel:
    """Callable wrapper (matches the model interface of write_cube)."""

    def __init__(self, cfg: OpenDRTConfig = MARC_CONFIG):
        self.cfg = cfg

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return opendrt_transform(rgb, self.cfg)
